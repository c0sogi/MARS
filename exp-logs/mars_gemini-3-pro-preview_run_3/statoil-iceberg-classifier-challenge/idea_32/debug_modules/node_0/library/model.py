import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module with Global Average Pooling.
    Acts as a low-pass filter to be robust against speckle noise.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        # Ensure hidden dimension is at least 1
        hidden_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class ResBlock(nn.Module):
    """
    Residual Block with specific stability constraints:
    - Bias=True in Convs
    - LeakyReLU
    - SE Module
    - Structure: Conv-BN-Act-Conv-BN-SE-Add-Act
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        use_bias = Config.USE_BIAS
        slope = Config.LEAKY_RELU_SLOPE

        # First convolution: handles stride for downsampling
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=use_bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=slope, inplace=True)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Hybrid Attention (SE Module)
        self.se = SEModule(out_channels)

        # Final activation after addition
        self.act2 = nn.LeakyReLU(negative_slope=slope, inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=use_bias,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.act2(out)
        return out


class BHAResNet(nn.Module):
    """
    Biased Hybrid-Attentive ResNet (BHA-ResNet).
    Custom 4-Stage ResNet optimized for small, noisy radar imagery.
    """

    def __init__(self):
        super(BHAResNet, self).__init__()

        # Load configuration
        stages = Config.MODEL_STAGES  # e.g., [64, 128, 128, 128]
        use_bias = Config.USE_BIAS
        slope = Config.LEAKY_RELU_SLOPE
        dropout_rate = Config.DROPOUT_RATE
        input_channels = Config.INPUT_CHANNELS

        # --- Stem ---
        # Initial feature extraction, keeping spatial resolution (75x75)
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stages[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
            nn.BatchNorm2d(stages[0]),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )

        # --- Backbone (4 Stages) ---
        # Aggressive downsampling: 75 -> 38 -> 19 -> 10 -> 5
        self.layer1 = ResBlock(stages[0], stages[0], stride=2)
        self.layer2 = ResBlock(stages[0], stages[1], stride=2)
        self.layer3 = ResBlock(stages[1], stages[2], stride=2)
        self.layer4 = ResBlock(stages[2], stages[3], stride=2)

        # --- Readout ---
        # Global Max Pooling to capture peak signals (icebergs)
        self.pool = nn.AdaptiveMaxPool2d(1)

        # --- Classification Head ---
        # Input dimension: Final stage channels + 1 (incidence angle)
        head_input_dim = stages[3] + 1
        hidden_dim = 256

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim, bias=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1, bias=True),
        )

        # Initialization: Using PyTorch defaults (Kaiming Uniform) as requested.

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image batch [B, 3, 75, 75]
            angle (torch.Tensor): Incidence angles [B]
        """
        # Feature Extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [B, C]

        # Feature Fusion
        # Ensure angle is [B, 1] for concatenation
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Concatenate raw angle with features
        x = torch.cat([x, angle], dim=1)

        # Classification
        logits = self.head(x)
        return logits
