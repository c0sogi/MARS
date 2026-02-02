import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Module.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Ensure hidden dimension is at least 1
        hidden_dim = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the Spatially-Regularized CNN.
    Sequence: Conv2d -> BN -> ReLU -> Dropout2d -> SE -> MaxPool2d.
    """

    def __init__(self, in_channels, out_channels, dropout_rate):
        super(ConvBlock, self).__init__()
        # Plain convolution (no residual) to enforce filtering
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Spatial Dropout drops entire feature maps
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.se = SELayer(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class SpatiallyRegularizedSECNN(nn.Module):
    """
    Custom 4-Stage Convolutional Network.
    Features:
    - Plain CNN backbone (aggressive downsampling).
    - Spatial Dropout for structural regularization.
    - Selective Hierarchical Pooling (Block 3 & 4).
    - Raw Scale Fusion with Incidence Angle.
    """

    def __init__(self):
        super(SpatiallyRegularizedSECNN, self).__init__()

        # Hyperparameters
        channels = Config.CHANNEL_SIZES  # Expected: [64, 128, 128, 128]
        spatial_dropout = Config.SPATIAL_DROPOUT_RATE
        head_dropout = Config.HEAD_DROPOUT_RATE

        # Backbone Stages
        self.block1 = ConvBlock(Config.IN_CHANNELS, channels[0], spatial_dropout)
        self.block2 = ConvBlock(channels[0], channels[1], spatial_dropout)
        self.block3 = ConvBlock(channels[1], channels[2], spatial_dropout)
        self.block4 = ConvBlock(channels[2], channels[3], spatial_dropout)

        # Global Pooling (Max Pooling for sparse, high-intensity signals)
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # Fusion Input: Block 3 Features + Block 4 Features + Incidence Angle
        fusion_dim = channels[2] + channels[3] + 1
        hidden_dim = 256  # Moderate capacity

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        PyTorch Default Initialization (Kaiming Uniform / Fan-In).
        Explicitly avoids fan_out or fixed scales.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, inc_angle):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Image input [Batch, 3, 75, 75]
            inc_angle (torch.Tensor): Incidence angle input [Batch]
        """
        # Stage 1
        x1 = self.block1(x)

        # Stage 2
        x2 = self.block2(x1)

        # Stage 3
        x3 = self.block3(x2)

        # Stage 4
        x4 = self.block4(x3)

        # Selective Pooling: Extract features from Block 3 and Block 4
        # Flatten: [Batch, C, 1, 1] -> [Batch, C]
        feat_3 = self.global_pool(x3).view(x3.size(0), -1)
        feat_4 = self.global_pool(x4).view(x4.size(0), -1)

        # Prepare angle for concatenation
        angle = inc_angle.view(-1, 1)

        # Fusion
        fused = torch.cat([feat_3, feat_4, angle], dim=1)

        # Classification
        logits = self.head(fused)

        return logits
