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
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    print('\nInit Model...', end=' ')

    # TODO
    # Create model and instantiate here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model_type == 'transformer':
        model = MILTransformer(input_size=384, hidden_size=256, num_classes=args.n_classes, device=device)
    elif args.model_type == 'max_pooling':
        model = MILModelMaxPooling(input_size=384, hidden_size=256, num_classes=args.n_classes, device=device)
    elif args.model_type == 'mean_pooling':
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
       optimizer, T_0=50, T_mult=2, eta_min=0.0005)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing,
                                        bs=1, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing, bs=1)
    test_loader = get_split_loader(test_split, testing = args.testing, bs=1)
    print('Done!')
    print('\nSetup EarlyStopping...', end=' ')
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
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, loss_fn, device = device)
        val_metrics   = validation_epoch(fold, epoch, model, val_loader, loss_fn, device = device)


        if max_roc_auc<=val_metrics['roc_auc']:
            max_roc_auc = val_metrics['roc_auc']
            #torch.save(model.state_dict(), os.path.join(args.results_dir, "{}/s_{}_{}_checkpoint.pt".format(fold, round(val_metrics['roc_auc'],3), fold)))

            test_metrics = summary(model, test_loader, device = device)
            wandb_update(train_metrics, val_metrics, test_metrics, scheduler)
        else:
            # Log separately if needed
            wandb_update(train_metrics, val_metrics, test_metrics=None, scheduler=scheduler)

    # Update the learning rate scheduler
    scheduler.step()
        
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
    return 0,0,0,0,0#results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_epoch(epoch, model, loader, optimizer, loss_fn = None, device = 'cpu') -> Dict[str, float]:

    metrics_logger = MetricsLogger()
    model.train()
   
    train_loss = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(tqdm(loader, desc=f'Training')):
        label = label.to(device)

        predictions = model(data)
        # Compute the loss
        loss = loss_fn(predictions, label)
        
        # Name a more iconic trio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update logger
        metrics_logger.update(predictions, label) 
        
        train_loss += loss.item()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    metrics = metrics_logger.compute_metrics() 
    # Print kmeans, silhouette and total loss without any inst loss
    print(f'Training Epoch: {epoch}: train_loss: {train_loss:.4f}')
    # Print the metrics
    print('#'*50)
    print(f'Metrics for training:\n{metrics}')
    

    return metrics


def validation_epoch(fold, epoch, model, loader, loss_fn = None, device = 'cpu') -> Dict[str, float]:
    metrics_logger = MetricsLogger()
    model.eval()

    val_loss = 0.

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(loader, desc='Validation')):
            label = label.to(device)

            predictions = model(data)
            # Compute the loss
            loss = loss_fn(predictions, label)
            # Update logger
            metrics_logger.update(predictions, label) 
            
            val_loss += loss.item()

    val_loss /= len(loader)
    metrics = metrics_logger.compute_metrics()

    print(f'Validation Epoch: {epoch}: val_loss: {val_loss:.4f}')
    print('#'*50)
    print(f'Metrics for validation:\n{metrics}')

    return metrics





def summary(model, loader, device = 'cpu') -> Dict[str, float]:
    model.eval()

    metrics_logger = MetricsLogger()

#    test_loss = 0.
    for batch_idx, (data, label) in enumerate(tqdm(loader, desc='Test')):
        label = label.to(device)

        ## see what this is used for
        #slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            predictions = model(data)

        metrics_logger.update(predictions, label)

    metrics = metrics_logger.compute_metrics()

    print('Test epoch')
    print('#'*50)
    print(f'Metrics for test set:\n{metrics}')

    return metrics


