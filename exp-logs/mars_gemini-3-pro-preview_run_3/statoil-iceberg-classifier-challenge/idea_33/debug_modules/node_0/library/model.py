import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class NonBottleneckSE(nn.Module):
    """
    Non-Bottleneck Squeeze-and-Excitation Module.

    Strategy:
    - Squeeze: Global Average Pooling to act as a low-pass filter robust to speckle noise.
    - Excitation: Full-rank MLP (No reduction ratio) to capture global dependencies
      between channels without an information bottleneck.
    """

    def __init__(self, channels: int):
        super(NonBottleneckSE, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Full-rank MLP: channels -> channels -> channels
        # No bottleneck (reduction ratio = 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Squeeze
        y = self.avg_pool(x).view(b, c)

        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        # Scale
        return x * y


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for Plain CNN.

    Structure:
    Conv2d (Bias=True) -> BatchNorm -> LeakyReLU -> NB-SE -> MaxPool
    """

    def __init__(self, in_channels: int, out_channels: int, slope: float = 0.1):
        super(ConvBlock, self).__init__()

        # Conv2d:
        # - Kernel size 3, padding 1 to maintain spatial dimensions before pooling.
        # - Bias=True explicitly retained to preserve initialization dynamics.
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

        self.bn = nn.BatchNorm2d(out_channels)

        # LeakyReLU: Preserves semantic negative values (e.g., shadows in radar).
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

        # Attention: Non-Bottleneck SE
        self.se = NonBottleneckSE(out_channels)

        # Downsampling: Max Pooling to capture high-intensity peaks.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class NBHACNN(nn.Module):
    """
    Non-Bottleneck Hybrid-Attentive Plain CNN (NBHA-CNN).

    Architecture:
    - 4-Stage Plain CNN Backbone
    - Non-Bottleneck Attention
    - Selective Hierarchical Max Pooling (Stage 3 & 4)
    - Raw Incidence Angle Fusion
    """

    def __init__(self, config: Config):
        super(NBHACNN, self).__init__()
        self.config = config

        # Hyperparameters
        widths = config.channel_widths  # Expected: [64, 128, 128, 128]
        slope = config.leaky_relu_slope
        in_c = config.input_channels

        # Backbone Stages
        # Sequential downsampling: 75 -> 37 -> 18 -> 9 -> 4
        self.stage1 = ConvBlock(in_c, widths[0], slope=slope)
        self.stage2 = ConvBlock(widths[0], widths[1], slope=slope)
        self.stage3 = ConvBlock(widths[1], widths[2], slope=slope)
        self.stage4 = ConvBlock(widths[2], widths[3], slope=slope)

        # Readout Strategy: Selective Hierarchical Max Pooling
        # We extract features from Stage 3 (index 2) and Stage 4 (index 3).
        dim_stage3 = widths[2]
        dim_stage4 = widths[3]
        dim_angle = 1

        # Total input dimension for the head
        readout_dim = dim_stage3 + dim_stage4 + dim_angle

        # Classification Head
        # Single hidden layer strategy
        hidden_dim = 256

        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        # Note: Weights are initialized using PyTorch default (Kaiming Uniform/Fan-In)
        # as per the design requirements.

    def forward(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input image tensor of shape (Batch, 3, 75, 75)
            angle: Incidence angle tensor of shape (Batch,) or (Batch, 1)

        Returns:
            logits: Raw output scores of shape (Batch, 1)
        """
        # Backbone Forward Pass
        f1 = self.stage1(x)  # Stage 1
        f2 = self.stage2(f1)  # Stage 2
        f3 = self.stage3(f2)  # Stage 3
        f4 = self.stage4(f3)  # Stage 4

        # Selective Hierarchical Max Pooling
        # Extract global context from Stage 3 and Stage 4
        # Global Max Pooling: (B, C, H, W) -> (B, C)
        p3 = F.adaptive_max_pool2d(f3, 1).view(f3.size(0), -1)
        p4 = F.adaptive_max_pool2d(f4, 1).view(f4.size(0), -1)

        # Angle Processing
        # Ensure angle is (B, 1) and on the correct device
        a = angle.view(-1, 1)

        # Feature Fusion
        # Concatenate pooled features and raw incidence angle
        features = torch.cat([p3, p4, a], dim=1)

        # Classification
        logits = self.head(features)

        return logits
