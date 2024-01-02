import os
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_auc_score, roc_curve, recall_score,
                        balanced_accuracy_score, confusion_matrix, precision_score,
                        f1_score)
from sklearn.metrics import auc as calc_auc
from tqdm import tqdm

from utils.utils import *
from utils.losses import compute_losses
from datasets.dataset_generic import save_splits

from models.model_mil import MIL_fc, MIL_fc_mc
from models.strat_models import multimodal_cluster
from models.model_clam import (CLAM_MB, CLAM_SB,
                    CLAM_MB_multimodal, CLAM_SB_multimodal, only_metadata_clam)


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')

    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')

    ### Change
    ### So it automatically changes the size of the features
    model_dict = {"dropout": args.drop_out,"dropout_value" : args.dropout,
        'n_classes': args.n_classes}

    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        if args.model_type =='clam_sb':
            #model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            model = CLAM_SB_multimodal(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB_multimodal(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)


    model = multimodal_cluster(model, hidden_layers= args.hidden_layers,
                            fus_method=args.fusion, dropout_value=args.dropout,
                            num_neurons=args.num_neurons,
                            final_hidden_layers=args.final_hidden_layers) 
    ## If we want only metadata
    #model = only_metadata_clam(model, hidden_layers=args.hidden_layers, dropout_value=args.dropout)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = CosineAnnealingWarmRestarts(
       optimizer, 50, T_mult=1, eta_min=0.0005)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing,
                                        bs=5, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing, bs=5)
    test_loader = get_split_loader(test_split, testing = args.testing, bs=5)
    print('Done!')
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 60, stop_epoch=80, verbose = True)

    else:
        print('No early stopping')
        early_stopping = None
    print('Done!')
     
    max_auc =-1.0

   #wandb.init(project=f'SLNB positivity {feat_type} 2023', entity='catai', reinit=True, config=args)
    model.to('cuda')
    print(args.max_epochs)
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
           #stop, metrics, losses = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
           #    early_stopping, writer, loss_fn, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop, metrics, losses = validate(cur, epoch, model, val_loader, args.n_classes,  early_stopping, writer, loss_fn, args.results_dir)
        continue
        ### CHANGE
        ### metrics[0] has the f1-score of the last val epoch
        if max_auc<=metrics[0]:
            max_auc = metrics[0]
            #torch.save(model.state_dict(), os.path.join(args.results_dir, "{}/s_{}_{}_checkpoint.pt".format(cur, round(metrics[0],3), cur)))
            wandb.log({"P-R curve" : wandb.plot.pr_curve(metrics[1], metrics[2], labels=['Negative', 'Positive'])})

            results_dict, test_error, test_auc, acc_logger, metrics_test = summary(model, test_loader, args.n_classes)
            
            wandb.log({"P-R curve test" : wandb.plot.pr_curve(metrics_test[1], metrics_test[2], labels=['Negative', 'Positive'])})

            # Log Val and Test at the same step!! 
            wandb.log({
                   'Val loss' : losses[0], 'Val F1' : metrics[0],
                   'Val error' : losses[1], 'Val inst loss' : losses[2],
                   'Val bal acc' : metrics[3], 'Val sensitivity': metrics[4],
                    'Val specificity': metrics[5], 'Val auc' : metrics[6],
                   'Val precision' : metrics[7],
                    'Test f1' : metrics_test[0],
                    'Test bal acc' : metrics_test[3],
                    'Test sensitivity': metrics_test[4],
                    'Test specificity': metrics_test[5],
                    'Test auc' : metrics_test[6],
                    'Test precision' : metrics_test[7],
                    'Best Val F1': max_auc}) # Shitty name as it is the ValF1 but we'll just leave it as it is for now
        else:
            # Log separately if needed
            wandb.log({
                   'Val loss' : losses[0], 'Val F1' : metrics[0],
                   'Val error' : losses[1], 'Val inst loss' : losses[2],
                   'Val bal acc' : metrics[3], 'Val sensitivity': metrics[4],
                   'Val specificity': metrics[5], 'Val auc' : metrics[6],
                   'Val precision' : metrics[7]})
        if stop: 
            break
        scheduler.step()
        
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

   #_, val_error, val_auc, _, _= summary(model, val_loader, args.n_classes)
   #print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

#   results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
#   print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

#   for i in range(args.n_classes):
#       acc, correct, count = acc_logger.get_summary(i)
#       print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

#       if writer:
#           writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

#   if writer:
#       writer.add_scalar('final/val_error', val_error, 0)
#       writer.add_scalar('final/val_auc', val_auc, 0)
#   #   writer.add_scalar('final/test_error', test_error, 0)
#   #   writer.add_scalar('final/test_auc', test_auc, 0)
#       writer.close()
    # TODO
    # remember to delete this as it stops our sweeps from running properly
    return 0,0,0,0,0#results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label, metadata) in tqdm(enumerate(loader)):
        label, metadata = label.to(device), metadata.to(device)

        distances, assignments, Y_hat, aggregated_features, instance_dict = model(data, label=label, instance_eval=True, metadata=metadata)


       #acc_logger.log(Y_hat, label)

        centroids = model.clustering_layer.centroids
        
        total_loss, loss_k_means, loss_silhouette = compute_losses(aggregated_features, centroids,
                                    assignments, k_means_loss_weight=1, silhouette_loss_weight=100)

        loss_value = total_loss.item()

      # instance_loss = instance_dict['instance_loss']
      # inst_count+=1
      # instance_loss_value = instance_loss.item()
      # train_inst_loss += instance_loss_value
      # 
      # total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

      # inst_preds = instance_dict['inst_preds']
      # inst_labels = instance_dict['inst_labels']
      # inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
     #  if (batch_idx + 1) % 20 == 0:
     #      print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
     #          'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        #error = calculate_error(Y_hat, label)
      # train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    # Print kmeans, silhouette and total loss without any inst loss
    print(f'Epoch: {epoch}: train_loss: {train_loss:.4f}, loss_k_means: {loss_k_means:.4f}, loss_silhouette: {loss_silhouette:.4f}')
  # if inst_count > 0:
  #     train_inst_loss /= inst_count
  #     print('\n')
  #     for i in range(2):
  #         acc, correct, count = inst_logger.get_summary(i)
  #         print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

  # print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
  # for i in range(n_classes):
  #     acc, correct, count = acc_logger.get_summary(i)
  #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
  #     if writer and acc is not None:
  #         writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

  # if writer:
  #     writer.add_scalar('train/loss', train_loss, epoch)
  #     writer.add_scalar('train/error', train_error, epoch)
  #     writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
  # wandb.log({
  #    'Train loss' : train_loss, 'Train AUC' : acc,
  #    'Train error' : train_error
  #     }, commit=False)


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    ### CHANGE
    Y_hats = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label, metadata) in enumerate(loader):
            data, label, metadata = data.to(device), label.to(device), metadata.to(device)
            distances, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True, metadata=metadata)

            acc_logger.log(Y_hat, label)
            loss = loss_fn(distances, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            Y_hats[batch_idx] = Y_hat.cpu().item()

            #error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        ### CHANGE
        ### We are giving these back to compute for the best model
        auc = roc_auc_score(labels, prob[:, 1])
        f1_value = f1_score(labels, Y_hats)
        bal_acc = balanced_accuracy_score(labels, Y_hats)
        sensitivity = recall_score(labels, Y_hats)
        cm = confusion_matrix(labels, Y_hats)
        specificity = cm[0][0]/(cm[0][0]+cm[0][1]) # TN/(TN+FP)
        precision   = precision_score(labels, Y_hats)
        metrics = (f1_value, labels, prob, bal_acc, sensitivity, specificity, auc, precision)
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, f1: {:.4f}'.format(val_loss, val_error, f1_value))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
    

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            ### CHANGE
            ### we are returning the metrics to save the model
            ### and compute the PR curve
            print("Early stopping")
            return True, metrics, [val_loss, val_error, val_inst_loss]
    
    ### CHANGE
    ### we are returning the metrics to save the model
    ### and compute the PR curve
    return False, metrics, [val_loss, val_error, val_inst_loss]

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    ### CHANGE
    ### We are adding this in order to compute other metrics later
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    Y_hats = np.zeros(len(loader))
    for batch_idx, (data, label, metadata) in enumerate(loader):
        data, label, metadata = data.to(device), label.to(device), metadata.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            distances, Y_prob, Y_hat, _, _ = model(data, metadata=metadata)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        ### CHANGE
        prob[batch_idx] = Y_prob.cpu().numpy()
        labels[batch_idx] = label.item()
        Y_hats[batch_idx] = Y_hat.cpu().item()

        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        #error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])

        f1_value = f1_score(labels, Y_hats)
        bal_acc = balanced_accuracy_score(labels, Y_hats)
        sensitivity = recall_score(labels, Y_hats)
        cm = confusion_matrix(labels, Y_hats)
        specificity = cm[0][0]/(cm[0][0]+cm[0][1]) #TN/(TN+FP)
        precision   = precision_score(labels, Y_hats)
        metrics = (f1_value, labels, prob, bal_acc, sensitivity, specificity, auc, precision)
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger, metrics


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label, metadata) in enumerate(loader):
        data, label, metadata= data.to(device), label.to(device), metadata.to(device)
        distances, Y_prob, Y_hat, _, _ = model.clam(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(distances, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        #error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    wandb.log({
       'Train loss' : train_loss, 'Train AUC' : acc,
       'Train error' : train_error
        }, commit=False)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    Y_hats = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label, metadata) in enumerate(loader):
            data, label, metadata= data.to(device), label.to(device), metadata.to(device)

            distances, Y_prob, Y_hat, _, _ = model(data, metadata=metadata)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(distances, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            Y_hats[batch_idx] = Y_hat.cpu().item()

            val_loss += loss.item()
            #error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        ### CHANGE
        ### We are giving these back to compute for the best model
        auc = roc_auc_score(labels, prob[:, 1])
        f1_value = f1_score(labels, Y_hats)
        bal_acc = balanced_accuracy_score(labels, Y_hats)
        sensitivity = recall_score(labels, Y_hats)
        cm = confusion_matrix(labels, Y_hats)
        specificity = cm[0][0]/(cm[0][0]+cm[0][1]) # TN/(TN+FP)
        precision   = precision_score(labels, Y_hats)
        metrics = (f1_value, labels, prob, bal_acc, sensitivity, specificity, auc, precision)
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
       
    # WANDB
    wandb.log({
       'Val loss' : val_loss, 'Val AUC' : auc,
       'Val error' : val_error
        })

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, f1: {:.4f}'.format(val_loss, val_error, f1_value))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False, metrics, [val_loss, val_error, 0]
