import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_clam import Fusion
from models.transformers import Transformer as transformer_module, Attn_Net_Gated

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

class MILTransformer(nn.Module):
    def __init__(self, input_size=384, hidden_size=256, num_classes=2, device='cpu', multitask=False):
        super(MILTransformer, self).__init__()
        self.multitask = multitask
        heads = 5
        dim = 32 * heads
        mlp_dim = dim
        self.attention = transformer_module(input_dim=input_size, dim=dim, mlp_dim=mlp_dim, heads=heads)
        self.device = device
        self.norm = nn.LayerNorm(dim)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size // 2, num_classes)
        )
        # Regression head for multitasking
        if self.multitask:
            self.regression_head = nn.Sequential(
                nn.Linear(dim, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_size // 2, 1),
                # As we have negative and positive values we do not apply a non-linearity
            )
        # Initialize weights
        initialize_weights(self)

    def forward(self, patches):
        try:
            aggregated_features = self.attention(patches)
        except Exception as e:
            print(patches.shape)
            raise e
        aggregated_features = self.norm(aggregated_features)
        # Classification prediction
        classification_pred = self.classification_head(aggregated_features)
        if self.multitask:
            # Regression prediction for multitask
            regression_pred = self.regression_head(aggregated_features)
            return classification_pred, regression_pred

        return classification_pred


class MILModelAtt(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=6, device= 'cpu'):
        super(MILModelAtt, self).__init__()
        self.attn_model = Attn_Net_Gated(L=input_size, D=hidden_size, dropout=True)
        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        #self.softmax = nn.Softmax(dim=1) # This is not needed as we use CrossEntropyLoss
        self.device = device

    def forward(self, x):
        # Apply the attention mechanism
        attn_weights, features = self.attn_model(x)
        attn_weights = attn_weights.squeeze()
        
        # Is this the right way to aggregate the features?
        aggregated_features = torch.sum(features * attn_weights.unsqueeze(-1), dim=1)
        x = torch.relu(self.fc1(aggregated_features))
        x = self.drop(x)
        x = self.fc2(x)
        # print("x: ", x.shape)
        return x

class MILModelMeanPooling(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=10, device= 'cpu'):
        super(MILModelMeanPooling, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        #self.softmax = nn.Softmax(dim=1) # This is not needed as we use CrossEntropyLoss
        self.device = device

    def forward(self, x):
        # Aggregate the features using mean operation
        aggregated_features = torch.mean(x, dim=1)  # Mean across the batch dimension
        x = torch.relu(self.fc1(aggregated_features))
        x = self.drop(x)
        x = self.fc2(x)
        return x # self.softmax(x)
    
class MILModelMaxPooling(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=10, device= 'cpu'):
        super(MILModelMaxPooling, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.device = device
        #self.softmax = nn.Softmax(dim=1) # This is not needed as we use CrossEntropyLoss

    def forward(self, x):
        # Aggregate the features using max operation
        aggregated_features, _ = torch.max(x, dim=1)  # Max across the batch dimension
        
        x = torch.relu(self.fc1(aggregated_features))
        x = self.drop(x)
        x = self.fc2(x)
        return x # self.softmax(x)

