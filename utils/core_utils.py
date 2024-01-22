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
from utils.losses import compute_losses, compute_clustered_nll
from datasets.dataset_generic import save_splits

from models.model_mil import MIL_fc, MIL_fc_mc
from models.strat_models import multimodal_cluster
from models.model_clam import (CLAM_MB, CLAM_SB,
                    CLAM_MB_multimodal, CLAM_SB_multimodal, only_metadata_clam)

from utils.metrics_strat import fit_nelson_aalen_clusters
from typing import Dict, Union

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
    c_index_results: Dict[str, float],
    test_c_index: float,
    test_loss: Dict[str, float],
    train_losses: Dict[str, float],
    val_losses: Dict[str, float],
    test_update: bool = False
) -> None:
    log_data: Dict[str, Union[float, Dict[str, float]]] = {
        'Val C index': c_index_results['c_index_val'],
        'Train C index': c_index_results['c_index_train'],
        'Train total loss': train_losses['total_loss'],
        'Train k_means loss': train_losses['k_means_loss'],
        'Train silhouette loss': train_losses['silhouette_loss'],
        'Val total loss': val_losses['total_loss'],
        'Val k_means loss': val_losses['k_means_loss'],
        'Val silhouette loss': val_losses['silhouette_loss']
    }

    if test_update:
        log_data.update({
            'Test total loss': test_loss['total_loss'],
            'Test k_means_loss': test_loss['k_means_loss'],
            'Test silhouette loss': test_loss['silhouette_loss'],
            'Test C index': test_c_index
        })

    wandb.log(log_data)


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))

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
                            final_hidden_layers=args.final_hidden_layers, temperature=0.2) 
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
                                        bs=40, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing, bs=40)
    test_loader = get_split_loader(test_split, testing = args.testing, bs=40)
    print('Done!')
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 60, stop_epoch=80, verbose = True)

    else:
        print('No early stopping')
        early_stopping = None
    print('Done!')
     
    max_c_index =-1.0

   #wandb.init(project=f'SLNB positivity {feat_type} 2023', entity='catai', reinit=True, config=args)
    wandb.init(project=f'Pruebas Clustering', entity='catai', reinit=True, config=args)
    model.to('cuda')
    for epoch in range(args.max_epochs):
        training_predictions, train_losses= train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight,  loss_fn)
        c_index_results, val_predictions, val_losses = validate_clam(cur, epoch,
                        model, val_loader, args.n_classes, training_predictions,  
                        early_stopping,  loss_fn, args.results_dir)

        #continue

        if max_c_index<=c_index_results['c_index_val']:
            max_c_index = c_index_results['c_index_val']
            #torch.save(model.state_dict(), os.path.join(args.results_dir, "{}/s_{}_{}_checkpoint.pt".format(cur, round(metrics[0],3), cur)))

            test_loss, _, _, _, test_c_index = summary(model, test_loader, args.n_classes, training_predictions)
            

            wandb_update(c_index_results, test_c_index, test_loss, train_losses, val_losses, test_update = True)
        else:
            # Log separately if needed
            # I want to pot c index and the losses for train and validation
            wandb_update(c_index_results, test_c_index=None, test_loss=None,
                           train_losses=train_losses, val_losses=val_losses)
       #if stop: 
       #    break
    scheduler.step()
        
   #if args.early_stopping:
   #    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
   #else:
   #    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

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


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
   
    # Initialize lists to store data for Nelson-Aalen estimator
    train_cluster_assignments = []
    train_survival_times = []    # Replace with actual survival t<Escape>imes data
    train_event_indicators = []  # Replace with actual event indicators data

    train_loss = 0.
    train_error = 0.
    total_kmeans_loss = 0.
    total_silhouette_loss = 0.

   #train_inst_loss = 0.
   #inst_count = 0

    print('\n')
    total_batches = len(loader)  # Get the total number of batches
    for batch_idx, (data, label, metadata) in enumerate(tqdm(loader, desc='Training')):
        label, metadata = label.to(device), metadata.to(device)

        distances, assignments, Y_hat, aggregated_features, instance_dict = model(data, label=label, instance_eval=True, metadata=metadata)


       #acc_logger.log(Y_hat, label)

        centroids = model.clustering_layer.centroids
        
        nll_loss = compute_clustered_nll(assignments, label)
        total_loss, loss_k_means, loss_silhouette = compute_losses(aggregated_features, centroids,
                                    assignments, k_means_loss_weight=1, silhouette_loss_weight=1)

        total_loss += nll_loss
        # Add cluster assignments to the list (assuming 'assignments' contains the cluster assignments)
        train_cluster_assignments.extend(torch.argmax(assignments, dim=1).detach().cpu().numpy())
        
        # Extract survival times and event indicators from label
        event_indicators = label[:, 0].detach().cpu().numpy()  # First column for event indicators
        survival_times = label[:, 1].detach().cpu().numpy()  # Second column for survival times
        train_survival_times.extend(survival_times)
        train_event_indicators.extend(event_indicators)

        # train_event_indicators.extend(extracted_event_indicators)import ipdb; ipdb.set_trace()

        loss_value = total_loss.item()

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss_value
        total_kmeans_loss += loss_k_means.item()
        total_silhouette_loss += loss_silhouette.item()

        # Unused for now as we are not using instance loss
      # instance_loss = instance_dict['instance_loss']
      # inst_count+=1
      # instance_loss_value = instance_loss.item()
      # train_inst_loss += instance_loss_value
      # 
      # total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

      # inst_preds = instance_dict['inst_preds']
      # inst_labels = instance_dict['inst_labels']
      # inst_logger.log_batch(inst_preds, inst_labels)

     #  if (batch_idx + 1) % 20 == 0:
     #      print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
     #          'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        #error = calculate_error(Y_hat, label)
      # train_error += error
        

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    total_kmeans_loss /= len(loader)
    total_silhouette_loss /= len(loader)

    # Print kmeans, silhouette and total loss without any inst loss
    print(f'Training Epoch: {epoch}: train_loss: {train_loss:.4f}, loss_k_means: {total_kmeans_loss:.4f}, loss_silhouette: {total_silhouette_loss:.4f}')



    # After training loop, prepare the data for Nelson-Aalen estimator
    train_data = {
        'clusters': np.array(train_cluster_assignments),
        'times': np.array(train_survival_times),
        'events': np.array(train_event_indicators)
    }

    losses_dict = {
        'total_loss': train_loss,
        'k_means_loss': total_kmeans_loss,
        'silhouette_loss': total_silhouette_loss,
        # Add other individual losses here
    }

# Return the updated values
    return train_data, losses_dict

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


def validate_clam(cur, epoch, model, loader, n_classes, training_predictions,  early_stopping = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_loss = 0.

    # Initialize lists to store data for Nelson-Aalen estimator
    val_cluster_assignments = []
    val_survival_times = []    # To store survival times
    val_event_indicators = []  # To store event indicators

#   val_inst_loss = 0. # Depracated
    total_kmeans_loss = 0.
    total_silhouette_loss = 0.

    ### CHANGE
    with torch.no_grad():
        for batch_idx, (data, label, metadata) in enumerate(tqdm(loader, desc='Validation')):
            label, metadata = label.to(device), metadata.to(device)

            distances, assignments, Y_hat, aggregated_features, instance_dict = model(data, label=label, instance_eval=True, metadata=metadata)

#           acc_logger.log(Y_hat, label)

            # Obtain centroids and compute losses
            centroids = model.clustering_layer.centroids
            total_loss, loss_k_means, loss_silhouette = compute_losses(aggregated_features, centroids,
                                        assignments, k_means_loss_weight=1, silhouette_loss_weight=1)

            loss_value = total_loss.item()

            val_loss += loss_value
            total_kmeans_loss += loss_k_means.item()
            total_silhouette_loss += loss_silhouette.item()

            # Add cluster assignments to the list
            val_cluster_assignments.extend(torch.argmax(assignments, dim=1).detach().cpu().numpy())

            # Extract survival times and event indicators from label
            survival_times = label[:, 1].detach().cpu().numpy()  # Second column for survival times
            event_indicators = label[:, 0].detach().cpu().numpy()  # First column for event indicators

            val_survival_times.extend(survival_times)
            val_event_indicators.extend(event_indicators)

    val_loss /= len(loader)
    total_kmeans_loss /= len(loader)
    total_silhouette_loss /= len(loader)

    

    # Prepare the data for Nelson-Aalen estimator
    val_data = {
        'clusters': np.array(val_cluster_assignments),
        'times': np.array(val_survival_times),
        'events': np.array(val_event_indicators)
    }

    losses_dict = {
        'total_loss': val_loss,
        'k_means_loss': total_kmeans_loss,
        'silhouette_loss': total_silhouette_loss,
    }

    # Compute the c-index and inverse Brier score
    # Note: Make sure the fit_nelson_aalen_clusters function is defined or imported
    c_index_results = fit_nelson_aalen_clusters(training_predictions, val_data)  # Using training_predictions here
           # The return looks like this, I wanna print it: return {'c_index_train': c_index_train, 'c_index_val': c_index_val}
    print(f'Validation Epoch: {epoch}: val_loss: {val_loss:.4f}, loss_k_means: {total_kmeans_loss:.4f}, loss_silhouette: {total_silhouette_loss:.4f}')
    print('#'*50)
    print(f'Metrics for training and validation:\n Train {c_index_results["c_index_train"]:.4f}, Validation {c_index_results["c_index_val"]:.4f}')

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
       #if early_stopping.early_stop:
       #    ### CHANGE
       #    ### we are returning the metrics to save the model
       #    ### and compute the PR curve
       #    print("Early stopping")
       #    return True, metrics, [val_loss, val_error, val_inst_loss]

    return c_index_results, val_data, losses_dict





def summary(model, loader, n_classes, training_predictions):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    test_loss = 0.
    total_kmeans_loss = 0.
    total_silhouette_loss = 0.

    cluster_assignments = []
    loader_survival_times = []    # To store survival times
    loader_event_indicators = []  # To store event indicators

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    ### CHANGE
    ### We are adding this in order to compute other metrics later
    prob = np.zeros((len(loader), n_classes))


    labels = np.zeros(len(loader))
    Y_hats = np.zeros(len(loader))
    for batch_idx, (data, label, metadata) in enumerate(tqdm(loader, desc='Test')):
        label, metadata = label.to(device), metadata.to(device)

        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            distances, assignments, Y_hat, aggregated_features, instance_dict = model(data, metadata=metadata)

        probs = Y_hat.cpu().numpy()
        #import ipdb; ipdb.set_trace()

        # Obtain centroids and compute losses
        centroids = model.clustering_layer.centroids
        total_loss, loss_k_means, loss_silhouette = compute_losses(aggregated_features, centroids,
                                    assignments, k_means_loss_weight=1, silhouette_loss_weight=1)

        loss_value = total_loss.item()

        test_loss += loss_value
        total_kmeans_loss += loss_k_means.item()
        total_silhouette_loss += loss_silhouette.item()

        cluster_assignments.extend(torch.argmax(assignments, dim=1).detach().cpu().numpy())
       
        # Extract survival times and event indicators from label
        survival_times = label[:, 1].detach().cpu().numpy()  # Second column for survival times
        event_indicators = label[:, 0].detach().cpu().numpy()  # First column for event indicators

        loader_survival_times.extend(survival_times)
        loader_event_indicators.extend(event_indicators)

  
    test_loss /= len(loader)
    total_kmeans_loss /= len(loader)
    total_silhouette_loss /= len(loader)

    loader_data = {
        'clusters': np.array(cluster_assignments),
        'times': np.array(loader_survival_times),
        'events': np.array(loader_event_indicators)
    }

    c_index_results = fit_nelson_aalen_clusters(training_predictions, loader_data)  # Using training_predictions here

    loss_dict = {
        'total_loss': test_loss,
        'k_means_loss': total_kmeans_loss,
        'silhouette_loss': total_silhouette_loss,
        }

    return loss_dict, prob, labels, Y_hats, c_index_results['c_index_val']


