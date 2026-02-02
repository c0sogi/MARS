import torch
import torch.nn as nn
from library.config import Config


class LocalShape1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for Local Trajectory Smoothing.

    This model takes a sequence of relative positions, dynamics, and signal metrics
    (in a local metric frame) and predicts the residual correction (Delta East, Delta North)
    for the center timestamp of the window.

    Architecture:
    1. Input: (Batch, Window_Size, Features)
    2. Backbone: Stack of 1D Conv layers with Batch Normalization and ReLU.
    3. Pooling: Global Average Pooling over the temporal dimension.
    4. Head: MLP regressing the 2D residual.
    """

    def __init__(self):
        super(LocalShape1DCNN, self).__init__()

        # Hyperparameters from Config
        self.cnn_channels = Config.CNN_CHANNELS
        self.kernel_size = Config.KERNEL_SIZE
        self.input_features = Config.NUM_FEATURES
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_rate = Config.DROPOUT

        # Build Convolutional Backbone
        layers = []
        in_channels = self.input_features

        for out_channels in self.cnn_channels:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,  # 'Same' padding to maintain length
                    bias=False,  # Bias is redundant with BatchNorm
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)

        # Prediction Head
        # Global Average Pooling reduces (B, C, L) -> (B, C)
        # So input dimension to MLP is the number of channels in the last Conv layer
        self.head = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(
                self.hidden_dim, 2
            ),  # Output: [Delta East, Delta North] in meters
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming Normal for Conv/Linear layers
        and constant initialization for BatchNorm.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Window_Size, Features).

        Returns:
            torch.Tensor: Predicted residuals of shape (Batch, 2).
        """
        # Permute input to (Batch, Features, Window_Size) for Conv1d
        x = x.transpose(1, 2)

        # Pass through Convolutional Backbone
        # Shape: (Batch, Last_Channel, Window_Size)
        x = self.backbone(x)

        # Global Average Pooling over temporal dimension
        # Shape: (Batch, Last_Channel)
        x = x.mean(dim=2)

        # Pass through MLP Head
        # Shape: (Batch, 2)
        x = self.head(x)

        return x
