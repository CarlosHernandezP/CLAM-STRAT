import os
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from typing import Dict, Union

from utils.utils import *
from datasets.dataset_generic import save_splits
from models.custom_mil_models import MILTransformer, MILModelMaxPooling, MILModelMeanPooling, MILModelAtt
from utils.metrics_strat import fit_nelson_aalen_clusters
from utils.eval_utils import MetricsLogger

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


def wandb_update(
    train_metrics : dict,
    val_metrics : dict,
    test_metrics : dict = None,
    scheduler: Union[None, CosineAnnealingWarmRestarts] = None
) -> None:

    log_data = {
        'Train roc_auc': train_metrics['roc_auc'],
        'Train f1_score': train_metrics['f1_score'],
        'Train precision': train_metrics['precision'],
        'Train recall': train_metrics['recall'],
        'Train balanced_accuracy': train_metrics['balanced_accuracy'],
        'Val roc_auc': val_metrics['roc_auc'],
        'Val f1_score': val_metrics['f1_score'],
        'Val precision': val_metrics['precision'],
        'Val recall': val_metrics['recall'],
        'Val balanced_accuracy': val_metrics['balanced_accuracy'],
        'Val_loss_braf': val_metrics['val_loss_braf'],
        'Val_loss_fga': val_metrics['val_loss_fga'] if 'val_loss_fga' in val_metrics.keys() else None,
        'Train loss braf': train_metrics['train_loss_braf'],
        'Train loss fga': train_metrics['train_loss_fga'] if 'train_loss_fga' in train_metrics.keys() else None,
        'Learning rate': scheduler.get_last_lr()[0] if scheduler else None
    }

    if test_metrics:
        log_data.update({
            'Test roc_auc': test_metrics['roc_auc'],
            'Test f1_score': test_metrics['f1_score'],
            'Test precision': test_metrics['precision'],
            'Test recall': test_metrics['recall'],
            'Test balanced_accuracy': test_metrics['balanced_accuracy'],
        })

    wandb.log(log_data)


def train(datasets, fold, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(fold))
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(fold)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    
    # TODO
    # Change to Binary Cross Entropy
    # Add MSE or smth if we add the FGA label
    loss_fn = []
    loss_fn.append(nn.CrossEntropyLoss())
    if args.use_fga:
       if args.model_type != 'transformer':
            raise ValueError('FGA label is only supported for the transformer model')
       loss_fn.append(nn.MSELoss())

    print('Done!')
    print('\nInit Model...', end=' ')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model_type == 'transformer':
        model = MILTransformer(input_size=384, hidden_size=256, num_classes=args.n_classes, device=device, multitask=args.use_fga)
    elif args.model_type == 'max':
        model = MILModelMaxPooling(input_size=384, hidden_size=256, num_classes=args.n_classes, device=device)
    elif args.model_type == 'mean':
        model = MILModelMeanPooling(input_size=384, hidden_size=256, num_classes=args.n_classes, device=device)
    elif args.model_type == 'attention':
        model = MILModelAtt(input_size=384, hidden_size=256, num_classes=args.n_classes, device=device)
    else:
        # Can you give a more descriptive error message?
        raise ValueError('Model type not recognized') 

    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)

    scheduler = CosineAnnealingWarmRestarts(
       optimizer, T_0=1, T_mult=2, eta_min=0.0005)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing,
                                        bs=1, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing, bs=1)
    test_loader = get_split_loader(test_split, testing = args.testing, bs=1)
    print('Done!')
    print('\nSetup EarlyStopping...', end=' ')

    ## TODO
    ## This is not functional yet
    if args.early_stopping:
        early_stopping = EarlyStopping(patience =60, stop_epoch=80, verbose = True)
    else:
        print('No early stopping')
        early_stopping = None
    print('Done!')
     
    max_roc_auc = 0 
    wandb.init(name = f'{args.model_type} - fga {args.use_fga}' , project=f'CVPR BRAF', entity='upc_gpi', reinit=True, config=args)

    model.to(device)
    for epoch in range(args.max_epochs):
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, loss_fn, device = device, multitask=args.use_fga)
        val_metrics   = validation_epoch(epoch, model, val_loader, loss_fn, device = device, multitask=args.use_fga)


        if max_roc_auc<=val_metrics['roc_auc']:
            max_roc_auc = val_metrics['roc_auc']
            #torch.save(model.state_dict(), os.path.join(args.results_dir, "{}/s_{}_{}_checkpoint.pt".format(fold, round(val_metrics['roc_auc'],3), fold)))

            test_metrics = summary(model, test_loader, device = device, multitask=args.use_fga)
            wandb_update(train_metrics, val_metrics, test_metrics,  scheduler)
        else:
            # Log separately if needed
            wandb_update(train_metrics, val_metrics, test_metrics=None, scheduler=scheduler)

        # Update the learning rate scheduler
        scheduler.step()

        # Early stopping
        if early_stopping:
            assert args.results_dir
            early_stopping(epoch, val_metrics['val_loss_braf'], model, ckpt_name = os.path.join(args.results_dir, f"s_{fold}_{args.model_type}_fga_{args.use_fga}_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            wandb.finish()
            return True  
   #if args.early_stopping:
   #    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(fold))))
   #else:
   #    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(fold)))

   #_, val_error, val_auc, _, _= summary(model, val_loader, args.n_classes)
   #print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

#   results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
#   print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    # TODO
    # remember to delete this as it stops our sweeps from running properly
    return True# 0,0,0,0,0#results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_epoch(epoch, model, loader, optimizer, loss_fn, device = 'cpu', multitask=False) -> Dict[str, float]:
    metrics_logger = MetricsLogger()
    model.train()
   
    train_loss_braf = 0.
    train_loss_fga = 0. if multitask else None

    for batch_idx, (data, labels) in enumerate(tqdm(loader, desc=f'Training')):
        data = data.to(device)
        if multitask:
            labels = (labels[0].to(device), labels[1].to(device).float()) # The float needs to be added
        else:
            labels = labels.to(device)

        # Predict from the model. If not multitask -> predictions is a tensor of shape (bs, n_classes)
        # If multitask -> predictions is a list of two tensors, one for each task

        predictions = model(data)

        if multitask:
            loss_braf = loss_fn[0](predictions[0], labels[0])
            # Compute the loss for FGA regression || We add the view so it does not throw an error
            loss_fga = loss_fn[1](predictions[1].view(1), labels[1])
            loss = loss_braf + loss_fga
            train_loss_fga += loss_fga.item()
            train_loss_braf += loss_braf.item()

            predictions, labels = predictions[0], labels[0]
        else:
        # Compute the loss for B-RAF classification
            loss_braf = loss_fn[0](predictions, labels)
            loss = loss_braf
            train_loss_braf += loss_braf.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics_logger.update(predictions, labels) 

    train_loss_braf /= len(loader)
    if multitask:
        train_loss_fga /= len(loader)

    metrics = metrics_logger.compute_metrics()

    # Add the losses to the metrics dictionary
    metrics['train_loss_braf'] = train_loss_braf
    if multitask:
        metrics['train_loss_fga'] = train_loss_fga

    # Print out the metrics
    print(f'Training Epoch: {epoch}:')
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}')
    print('#' * 50)
    return metrics

def validation_epoch(epoch, model, loader, loss_fn, device='cpu', multitask=False) -> Dict[str, float]:
    metrics_logger = MetricsLogger()
    model.eval()

    val_loss_braf = 0.
    val_loss_fga = 0. if multitask else None

    with torch.no_grad():
        for _, (data, labels) in enumerate(tqdm(loader, desc='Validation')):
            data = data.to(device)
            if multitask:
                labels = (labels[0].to(device), labels[1].to(device).float())
            else:
                labels = labels.to(device)

            predictions = model(data)

            if multitask:
                # Compute the loss for B-RAF classification
                loss_braf = loss_fn[0](predictions[0], labels[0] if multitask else labels) # Maybe we can clean the code if we do this if statements
                loss_fga = loss_fn[1](predictions[1].view(1), labels[1])

                val_loss_braf += loss_braf.item()
                val_loss_fga += loss_fga.item()
                predictions, labels = predictions[0], labels[0]
            else:
                loss_braf = loss_fn[0](predictions, labels)
                val_loss_braf += loss_braf.item()

            metrics_logger.update(predictions, labels)

    val_loss_braf /= len(loader)
    if multitask:
        val_loss_fga /= len(loader)
    metrics = metrics_logger.compute_metrics()
    # Add the losses to the metrics dictionary
    metrics['val_loss_braf'] = val_loss_braf
    if multitask:
        metrics['val_loss_fga'] = val_loss_fga

    # Print out the metrics
    print(f'Validation Epoch: {epoch}:')
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}')
    print('#' * 50)
    return metrics


def summary(model, loader, device = 'cpu', multitask=False) -> Dict[str, float]:
    model.eval()

    metrics_logger = MetricsLogger()

#    test_loss = 0.
    for batch_idx, (data, label) in enumerate(tqdm(loader, desc='Test')):
        data = data.to(device)
        if multitask:
            label = (label[0].to(device), label[1].to(device))
        else:
            label = label.to(device)
        ## see what this is used for
        #slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            predictions = model(data)
        if multitask:
            predictions, label = predictions[0], label[0]

        metrics_logger.update(predictions, label)

    metrics = metrics_logger.compute_metrics()

    print('Test epoch')
    print('#'*50)
    print(f'Metrics for test set:\n{metrics}')

    return metrics


