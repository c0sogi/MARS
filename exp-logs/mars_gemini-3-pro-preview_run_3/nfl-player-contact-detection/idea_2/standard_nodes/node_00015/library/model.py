import torch
import torch.nn as nn
from library.config import Config


class TemporalCNN(nn.Module):
    """
    Temporal CNN for NFL Contact Detection.
    Processes temporal windows of player tracking features.
    """

    def __init__(self):
        super(TemporalCNN, self).__init__()

        self.in_channels = Config.NUM_FEATURES
        self.hidden_channels = Config.HIDDEN_CHANNELS
        self.kernel_size = Config.KERNEL_SIZE
        self.dropout_p = Config.DROPOUT

        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(self.hidden_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(self.hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Time).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x
