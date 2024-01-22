import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_clam import Fusion
from utils.utils import initialize_weights

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class multimodal_cluster(nn.Module):
    def __init__(self, clam, fus_method='cat',
                    dropout_value=0.2, hidden_layers=3, final_hidden_layers=2, num_neurons = 32,
                 num_clusters: int = 5, temperature: float = 1.0) -> None:
        super(multimodal_cluster, self).__init__()

        self.clam = clam
        self.clam_type= 'sb'
        self.temperature = temperature
        if fus_method=='cat':
            layers = [nn.Linear(3, num_neurons), nn.ReLU()]
            for _ in range(hidden_layers):
                layers.extend([nn.Linear(num_neurons, num_neurons), nn.ReLU()])
            self.metadata_head = nn.Sequential(*layers)

            # WHY ARE WE HARD CODING THIS?
            num_feats = 256+num_neurons
        else:
            num_feats=64
            self.metadata_head = nn.Sequential(
                            nn.Linear(3, 32),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_value),
                            nn.Linear(32, 64),
                            nn.ReLU())
            
        #self.instance_norm_patch = nn.InstanceNorm1d(
        # ------------- Data fusion & Clustering --------------------- 
        self.fus_method = fus_method
        self.fusion = Fusion(method = fus_method, clam_type=self.clam_type)
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.instance_norm_patches = nn.InstanceNorm1d(64)
        self.instance_norm_meta = nn.InstanceNorm1d(num_neurons)
     
        # Cluster specific
        self.num_clusters = num_clusters
        self.clustering_layer = ClusteringLayer(self.num_clusters, feature_dim=num_feats, temperature =self.temperature)
        
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False,
                                                    attention_only=False, metadata=None):
        batch_distances = []
        batch_Y_prob = []
        batch_Y_hat = []
        batch_aggregated_info = []
        #batch_A_raw = []
        batch_results_dict = []

        for h_single, metadata_single in zip(h, metadata):
            metadata_single = metadata_single.unsqueeze(0)
            h_single = h_single.to(self.device)
            
            M, total_inst_loss, all_targets, all_preds, A_raw = self.clam(h_single, label, instance_eval)
            M = self.instance_norm_patches(M.unsqueeze(dim=0)).squeeze(dim=0)
            
            metadata_single = self.instance_norm_meta(self.metadata_head(metadata_single).unsqueeze(dim=0)).squeeze(dim=0)
            
            aggregated_info = self.fusion(M, metadata_single)
            
            distances = self.clustering_layer(aggregated_info)
            Y_hat = torch.topk(-distances, 1, dim=1)[1]
            Y_prob = F.softmax(-distances / self.temperature, dim=1)
            
            if instance_eval:
                results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
                                'inst_preds': np.array(all_preds)}
            else:
                results_dict = {}
            batch_distances.append(distances)
            batch_Y_prob.append(Y_prob)
            batch_Y_hat.append(Y_hat)
            batch_aggregated_info.append(aggregated_info)
            #batch_A_raw.append(A_raw)
            batch_results_dict.append(results_dict)

        # Concatenate all the tensors along the batch dimension
        distances = torch.cat(batch_distances, dim=0)
        Y_prob = torch.cat(batch_Y_prob, dim=0)
        Y_hat = torch.cat(batch_Y_hat, dim=0)
        aggregated_info = torch.cat(batch_aggregated_info, dim=0)
       #A_raw = torch.cat(batch_A_raw, dim=0)

        return distances, Y_prob, Y_hat, aggregated_info,  batch_results_dict   # A_raw, has been eliminated        




class multimodalHead_cluster(nn.Module):
    def __init__(self, feature_dim, num_heads, fus_method='cat',
                 hidden_layers=3, num_neurons=32, num_clusters: int = 5, temperature: float = 1.0) -> None:
        super(multimodal_cluster, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.temperature = temperature
        
        self.metadata_head = self._make_layers(3, num_neurons, hidden_layers)
        self.fus_method = fus_method
        self.fusion = Fusion(method=fus_method)
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        num_feats = feature_dim + num_neurons
        self.instance_norm_patches = nn.InstanceNorm1d(num_feats)
        self.instance_norm_meta = nn.InstanceNorm1d(num_neurons)
        self.clustering_layer = ClusteringLayer(num_clusters, feature_dim=num_feats, temperature=self.temperature)
        initialize_weights(self)

    def _make_layers(self, in_features, out_features, num_layers):
        layers = [nn.Linear(in_features, out_features), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(out_features, out_features), nn.ReLU(inplace=True)])
        return nn.Sequential(*layers)

    def forward(self, h, metadata):
        metadata_processed = self.metadata_head(metadata)
        metadata_processed = self.instance_norm_meta(metadata_processed)
        
        h = h.to(self.device).transpose(0, 1)  # (L, N, E) format for multi-head attention
        attn_output, _ = self.multihead_attention(h, h, h)
        attn_output = attn_output.transpose(0, 1).mean(dim=1)  # Aggregate over the sequence length dimension
        attn_output = self.instance_norm_patches(attn_output)

        aggregated_info = self.fusion(attn_output, metadata_processed)
        distances = self.clustering_layer(aggregated_info)
        Y_hat = torch.topk(-distances, 1, dim=1)[1]
        Y_prob = F.softmax(-distances / self.temperature, dim=1)

        return distances, Y_prob, Y_hat, aggregated_info



class ClusteringLayer(nn.Module):
    def __init__(self, num_clusters, feature_dim, temperature=1.0):
        super(ClusteringLayer, self).__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.centroids = nn.Parameter(torch.randn(num_clusters, feature_dim))  # Initialize cluster centroids

    def forward(self, x):
        # Compute squared L2 distance between data points and centroids
        distances = torch.sum((x.unsqueeze(1) - self.centroids)**2, dim=2)
        return distances

