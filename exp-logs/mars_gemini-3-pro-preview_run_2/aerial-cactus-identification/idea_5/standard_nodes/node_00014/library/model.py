import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Adaptively recalibrates channel-wise feature responses.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Ensure hidden dimension is at least 1
        hidden_channels = max(1, channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Learn channel weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Reweight feature maps
        return x * y


class BasicBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Uses two 3x3 convolutions and an SE block.
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(BasicBlock, self).__init__()

        # First convolution
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

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE Module
        self.se = SEBlock(out_channels, reduction)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply SE attention
        out = self.se(out)

        # Residual connection
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CustomSEResNet(nn.Module):
    """
    Custom Lightweight SE-ResNet designed for 32x32 images.
    Features:
    - 3x3 Convolutions exclusively.
    - Controlled downsampling to maintain 8x8 feature map at the end.
    - Squeeze-and-Excitation blocks.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_channels=32,
        layers=[2, 2, 2],
        strides=[1, 2, 2],
        se_reduction=16,
        dropout=0.0,
    ):
        super(CustomSEResNet, self).__init__()

        self.in_channels = base_channels

        # Initial Stem: 3x3 conv, stride 1 (Input 32x32 -> Output 32x32)
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # Build Stages dynamically based on config
        self.stages = nn.ModuleList()

        for i, (num_blocks, stride) in enumerate(zip(layers, strides)):
            # Calculate output channels: base * 2^i
            # Stage 0: 32 channels
            # Stage 1: 64 channels
            # Stage 2: 128 channels
            out_channels = base_channels * (2**i)

            stage = self._make_layer(out_channels, num_blocks, stride, se_reduction)
            self.stages.append(stage)

        # Classification Head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.in_channels, num_classes)

        # Weight Initialization
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride, reduction):
        layers = []
        # First block handles stride and channel expansion
        layers.append(BasicBlock(self.in_channels, out_channels, stride, reduction))
        self.in_channels = out_channels

        # Subsequent blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, reduction))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))

        # Stages
        for stage in self.stages:
            x = stage(x)

        # Head
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
