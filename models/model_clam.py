import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 2048, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):

        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 2048, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, feat_type='simclr'):
        super(CLAM_SB, self).__init__()
        if feat_type=='simclr':
            self.size_dict = {"small": [2048, 128, 64], "big": [1024, 512, 384]}
        else:
            self.size_dict = {"small": [1024, 128, 64], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        #self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device

        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB_multimodal(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small",
                 dropout = False, k_sample=8, n_classes=2, dropout_value=0.2,
        instance_loss_fn=nn.CrossEntropyLoss(), feat_type='simclr', subtyping=False):

        nn.Module.__init__(self)
        if feat_type=='simclr':
            self.size_dict = {"small": [384, 128, 128], "big": [1024, 512, 384]}

        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout_value))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        
        device = h.device
        A, h = self.attention_net(h)  # NxK        
         
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        else:
            total_inst_loss = 0.0
            all_targets = []
            all_preds = []
        M = torch.mm(A, h) 
        return M, total_inst_loss, all_targets, all_preds, A_raw


class CLAM_SB_multimodal(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small",
                 dropout = False, k_sample=8, n_classes=2, dropout_value=0.2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, feat_type='simclr'):

        nn.Module.__init__(self)
        self.size_dict = {"small": [384, 256, 128], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout_value))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        
       #if instance_eval:
       #    total_inst_loss = 0.0
       #    all_preds = []
       #    all_targets = []
       #    inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
       #    for i in range(len(self.instance_classifiers)):
       #        inst_label = inst_labels[i].item()
       #        classifier = self.instance_classifiers[i]
       #        if inst_label == 1: #in-the-class:
       #            instance_loss, preds, targets = self.inst_eval(A, h, classifier)
       #            all_preds.extend(preds.cpu().numpy())
       #            all_targets.extend(targets.cpu().numpy())
       #        else: #out-of-the-class
       #            if self.subtyping:
       #                instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
       #                all_preds.extend(preds.cpu().numpy())
       #                all_targets.extend(targets.cpu().numpy())
       #            else:
       #                continue
       #        total_inst_loss += instance_loss

       #    if self.subtyping:
       #        total_inst_loss /= len(self.instance_classifiers)
       #else:
       #    total_inst_loss = 0.0
       #    all_targets = []
       #    all_preds = []
                
        M = torch.mm(A, h) 
        total_inst_loss, all_targets, all_preds = 0, [], []
        return M, total_inst_loss, all_targets, all_preds, A_raw

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), feat_type='simclr', subtyping=False):
        nn.Module.__init__(self)
        if feat_type=='simclr':
            self.size_dict = {"small": [2048, 128, 64], "big": [1024, 512, 384]}
        else:
            self.size_dict = {"small": [1024, 128, 64], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.40))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):

        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict



class Fusion(nn.Module):
    "Multimodal data aggregator."
    def __init__(self, method, clam_type):
        super(Fusion, self).__init__()
        self.method = method
        self.clam_type = clam_type
        methods = ['cat', 'max', 'sum', 'prod']

        if self.method not in methods:
            raise ValueError('"method" must be one of ', methods)
    def _single_branch(self, patches, metadata):
        if self.method == 'cat':
            out = torch.cat([patches, metadata], dim=1)
        else:
            x_0 = torch.cat([patches, metadata], dim=0)
            if self.method == 'max':
                out = x_0.max(dim=0)[0].unsqueeze(dim=0)
            if self.method == 'sum':
                out = x_0.sum(dim=0).unsqueeze(dim=0)
            if self.method == 'prod':
                out = x_0.prod(dim=0).unsqueeze(dim=0)
        return out

    def _multi_branch(self, patches, metadata):
        if self.method == 'cat':
            M1 = torch.cat([patches[0].unsqueeze(dim=0), metadata], dim=1)
            M2 = torch.cat([patches[1].unsqueeze(dim=0), metadata], dim=1)
            out = torch.cat([M1, M2], dim=0)
        else:
            x_0 = torch.cat([patches[0].unsqueeze(dim=0), metadata], dim=0)
            x_1 = torch.cat([patches[1].unsqueeze(dim=0), metadata], dim=0)
            if self.method == 'max':
                out_0 = x_0.max(dim=0)[0].unsqueeze(dim=0)
                out_1 = x_1.max(dim=0)[0].unsqueeze(dim=0)
                out = torch.cat([out_0, out_1], dim=0)
            if self.method == 'sum':
                out_0 = x_0.sum(dim=0).unsqueeze(dim=0)
                out_1 = x_1.sum(dim=0).unsqueeze(dim=0)
                out = torch.cat([out_0, out_1], dim=0)
            if self.method == 'prod':
                out_0 = x_0.prod(dim=0).unsqueeze(dim=0)
                out_1 = x_1.prod(dim=0).unsqueeze(dim=0)
                out = torch.cat([out_0, out_1], dim=0)
        return out

    def forward(self, patches, metadata):
        if self.clam_type == 'sb':
            out = self._single_branch(patches, metadata)
        else:
            out = self._multi_branch(patches, metadata)
        return out

class only_metadata_clam(nn.Module):
    def __init__(self, clam, fus_method='cat', dropout_value=0.2, hidden_layers=3):
        super(only_metadata_clam, self).__init__()
        self.clam = None
        
        num_feats = 64
        if hidden_layers==3:
            self.metadata_head = nn.Sequential(
                            nn.Linear(3, 32),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_value),
                            nn.Linear(32, 32),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_value),
                            nn.Linear(32, num_feats),
                            nn.ReLU())
        if hidden_layers ==2:
            self.metadata_head = nn.Sequential(
                            nn.Linear(3, 32),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_value),
                            nn.Linear(32, num_feats),
                            nn.ReLU())
        if hidden_layers ==1:
            num_feats=32
            self.metadata_head = nn.Sequential(
                            nn.Linear(3, num_feats),
                            nn.ReLU())


        self.n_classes = 1
        self.classifiers = nn.Linear(num_feats, 2)
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False, metadata=None):

        metadata = self.metadata_head(metadata)
        logits = self.classifiers(metadata)

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {}
        A_raw = []
        if instance_eval:
            results_dict = {'instance_loss': 0, 'inst_labels': 0, 
            'inst_preds': 0}
        else: results_dict = {}
        return logits, Y_prob, Y_hat, A_raw,  results_dict
