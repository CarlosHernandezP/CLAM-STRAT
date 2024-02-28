import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_clam import Fusion
from models.transformers import Transformer as transformer_module

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

class MIL_transformer(nn.Module):
    def __init__(self, input_size=384,
                        hidden_size=256, num_classes=2) -> None:
        
        # To aggregate features
        self.attention = transformer_module(dim=256, mlp_dim=256)

        # Create a sequential with these layers and relus in between
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size // 2, num_classes)
        )
        # Cluster specific
        initialize_weights(self)
        
    def forward(self, patches, attention_only=False):
        batch_predictions = []

        for x_single in patches:
            x_single = x_single.to(self.device)
            
            # Aggregate features
            try:
                aggregated_features = self.attention(x_single.unsqueeze(0))
            except:
                print(x_single.shape)
                raise
            # Obtain a prediction from the aggregated features 
            pred = self.mlp(aggregated_features)
            # Combine so as to be able to use multiple WSI per batch
            batch_predictions.append(pred)
        pred = torch.cat(batch_predictions, dim=0)
        if attention_only:
            raise NotImplementedError
        return pred
