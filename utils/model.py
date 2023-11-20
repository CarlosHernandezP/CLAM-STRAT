import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        
        self.encoder = models.resnet50(pretrained= True)
        dim_in = self.encoder.fc.in_features
        

        self.encoder.fc = nn.Identity()

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class SupCEEffNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, num_classes=2, model_name='effb0'):
        super(SupCEEffNet, self).__init__()

        self.encoder = choose_model(model_name=model_name)
        dim_in = self.encoder._fc.in_features
        self.encoder._fc = nn.Identity()
        self.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(dim_in, int(dim_in/2)),
                Swish_Module(),
                nn.Dropout(p=0.4),
                nn.Linear(int(dim_in/2), num_classes))

    def forward(self, x):
        feat = self.encoder(x)
        
        return self.fc(feat)


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, num_classes=2, model_name='resnet50'):
        super(SupCEResNet, self).__init__()
        self.encoder = choose_model(model_name=model_name)
        dim_in = self.encoder.fc.in_features

        self.encoder.fc = nn.Identity()
        self.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(dim_in, int(dim_in/2)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(int(dim_in/2), num_classes))

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(self.encoder(x))

class SupConEffNet(nn.Module):
    def __init__(self, head='mlp', feat_dim=128):
        super(SupConEffNet, self).__init__()
        
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        dim_in = self.encoder._fc.in_features
        self.encoder._fc = nn.Identity()
        self.encoder._swish = nn.Identity()

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self,feat_dim = 2048, num_classes=2, classifier='linear'):
        super(LinearClassifier, self).__init__()
        feat_dim = feat_dim 
        if classifier == 'linear':
            self.fc = nn.Linear(feat_dim, num_classes)
        elif classifier == 'mlp':
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim/2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim/2, num_classes))


    def forward(self, features):
        
        return self.fc(features)

def choose_model(model_name : str) -> nn.Module:
    if 'res' in model_name:
        if '18' in model_name:
            feature_extractor = models.resnet18(pretrained=True)
        elif '34' in model_name:
            feature_extractor = models.resnet34(pretrained=True)
        elif '50' in model_name:
            feature_extractor = models.resnet50(pretrained=True)
        elif '101' in model_name:
            feature_extractor = models.resnet101(pretrained=True)
        else:
            raise NotImplementedError("The feature extractor cannot be instantiated: model asked -> {} does not exist".format(model_name))

    elif 'eff' in model_name:
        if 'b0' in model_name:
            feature_extractor = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
        elif 'b1' in model_name:
            feature_extractor = EfficientNet.from_pretrained('efficientnet-b1', num_classes=2)
        elif 'b2' in model_name:
            feature_extractor = EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
        elif 'b3' in model_name:
            feature_extractor = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
        else:
            raise NotImplementedError("The feature extractor cannot be instantiated: model asked -> {} does not exist".format(model_name))
    else:
        raise NotImplementedError("The feature extractor cannot be instantiated: model asked -> {} does not exist".format(model_name))
    
    return feature_extractor

sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod 
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
