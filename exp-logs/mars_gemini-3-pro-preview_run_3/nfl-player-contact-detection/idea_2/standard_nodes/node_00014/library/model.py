import torch
import torch.nn as nn
from library.config import Config


class ContactMLP(nn.Module):
    """
    Multi-Layer Perceptron for NFL Contact Detection.
    Processes flattened temporal windows of player tracking features.
    """

    def __init__(self):
        super(ContactMLP, self).__init__()

        # Input dimension: Features * Window Size
        self.input_dim = Config.NUM_FEATURES * Config.WINDOW_SIZE
        self.layers_dim = Config.MLP_LAYERS
        self.dropout_p = Config.DROPOUT

        layers = []
        in_dim = self.input_dim

        for out_dim in self.layers_dim:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_p))
            in_dim = out_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer
        self.fc_out = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Input_Dim).
        """
        x = self.feature_extractor(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x
