import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module.

    Implementation Details:
    - Uses Global Average Pooling for the 'Squeeze' operation to act as a low-pass filter
      for channel statistics, preventing overfitting to local noise (Lesson 104).
    - Standard reduction ratio of 16 (implied by 'Standard SE Module').
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class HybridSECNN(nn.Module):
    """
    Hybrid-Pooling SE-CNN Architecture.

    Key Features:
    - Backbone: 4-Stage Plain CNN (Conv-BN-Leaky-SE-Pool).
    - Hybrid Pooling: AvgPool for SE Attention, MaxPool for Classification Features.
    - Activation: LeakyReLU to preserve negative signal values.
    - Fusion: Concatenates raw incidence angle with image features.
    """

    def __init__(self):
        super(HybridSECNN, self).__init__()

        layers = []
        in_channels = Config.IN_CHANNELS

        # --- Backbone Construction ---
        # Strategy: Early expansion to 128 channels, then capped.
        # 4 Blocks: 75x75 -> 37x37 -> 18x18 -> 9x9 -> 4x4
        for out_channels in Config.BLOCK_CHANNELS:
            block = nn.Sequential(
                # Conv: Retain bias for initialization dynamics
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=Config.USE_BIAS,
                ),
                nn.BatchNorm2d(out_channels),
                # Activation: LeakyReLU to preserve semantic negative values (radar shadows)
                nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
                # Attention: SE Module (Avg Pooling)
                SEModule(out_channels),
                # Downsampling: MaxPool
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            layers.append(block)
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # --- Classification Head ---
        # Input features: Final Conv Channels + 1 (Incidence Angle)
        final_conv_channels = Config.BLOCK_CHANNELS[-1]
        self.head_in_features = final_conv_channels + 1

        # Hidden layer dimension (e.g., 256)
        hidden_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(self.head_in_features, hidden_dim),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialization: PyTorch Default (Kaiming Uniform / Fan-In).
        Explicitly applied to ensure consistency.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                    a=Config.LEAKY_RELU_SLOPE,
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                    a=Config.LEAKY_RELU_SLOPE,
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        """
        Forward pass with feature fusion.

        Args:
            x (torch.Tensor): Image input of shape (Batch, 3, 75, 75).
            angle (torch.Tensor): Incidence angle input of shape (Batch,).
        """
        # 1. Feature Extraction
        x = self.features(x)

        # 2. Global Max Pooling for Classification
        # Captures peak signals (iceberg signatures)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten -> (Batch, 128)

        # 3. Feature Fusion
        # Concatenate raw angle: (Batch, 128) + (Batch, 1) -> (Batch, 129)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        # 4. Classification
        x = self.classifier(x)
        return x
