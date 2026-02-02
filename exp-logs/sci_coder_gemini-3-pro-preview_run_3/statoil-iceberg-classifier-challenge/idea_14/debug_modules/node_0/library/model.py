import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MODEL_PARAMS


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with SE attention and optional downsampling.
    """

    def __init__(self, in_channels, out_channels, stride=1, se_reduction=16):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=se_reduction)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class HybridWideSEResNet(nn.Module):
    """
    Custom 4-Stage Residual Network with SE blocks, Global Max Pooling,
    and Raw-Scale Incidence Angle Fusion.
    """

    def __init__(self):
        super(HybridWideSEResNet, self).__init__()

        # Load configuration
        in_channels = MODEL_PARAMS["input_channels"]
        stem_channels = MODEL_PARAMS["stem_channels"]
        block_channels = MODEL_PARAMS["block_channels"]  # Expected: [64, 128, 128, 128]
        se_reduction = MODEL_PARAMS["se_reduction"]
        dropout_rate = MODEL_PARAMS["dropout_rate"]
        self.use_angle = MODEL_PARAMS["use_angle"]

        # 1. Stem
        # Initial processing, keeping stride 1 to preserve spatial dims before stages
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Backbone Stages (4 Downsampling operations total)
        # We apply stride=2 in every stage to achieve ~5x5 output from 75x75 input
        self.layer1 = self._make_layer(
            stem_channels, block_channels[0], stride=2, se_reduction=se_reduction
        )
        self.layer2 = self._make_layer(
            block_channels[0], block_channels[1], stride=2, se_reduction=se_reduction
        )
        self.layer3 = self._make_layer(
            block_channels[1], block_channels[2], stride=2, se_reduction=se_reduction
        )
        self.layer4 = self._make_layer(
            block_channels[2], block_channels[3], stride=2, se_reduction=se_reduction
        )

        # 3. Global Max Pooling (Preserve peak signals)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        # 4. Classification Head with Fusion
        # Calculate input dimension for the linear layer
        feature_dim = block_channels[-1]
        if self.use_angle:
            feature_dim += 1  # Add scalar angle

        hidden_dim = 256

        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        # Weight Initialization: Relying on PyTorch defaults (Kaiming Uniform)
        # as explicitly requested ("Strictly use PyTorch Default Initialization").

    def _make_layer(self, in_channels, out_channels, stride, se_reduction):
        # Constructs a stage. Given the small dataset and "Simple CNN" constraints,
        # we use 1 block per stage which performs the downsampling/expansion.
        return ResidualBlock(
            in_channels, out_channels, stride=stride, se_reduction=se_reduction
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image input of shape (B, 3, 75, 75)
            angle (torch.Tensor): Incidence angle input of shape (B,)
        """
        # Backbone
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (B, 128)

        # Fusion
        if self.use_angle:
            # Reshape angle to (B, 1) for concatenation
            angle = angle.view(-1, 1)
            # Concatenate raw angle (no normalization)
            x = torch.cat((x, angle), dim=1)

        # Classification
        x = self.head(x)
        return x
