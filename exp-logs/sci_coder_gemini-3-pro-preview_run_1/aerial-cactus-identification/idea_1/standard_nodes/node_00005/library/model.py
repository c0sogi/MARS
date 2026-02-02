import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A Residual Block composed of two 3x3 convolutions with Batch Normalization
    and ReLU activations. Includes a skip connection that adapts dimensions
    via a 1x1 convolution if necessary.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # First convolution: 3x3, handles stride for downsampling
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

        # Second convolution: 3x3, stride 1, maintains dimensions
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # If input and output shapes do not match (due to stride or channel change),
        # use a 1x1 convolution to project input to output shape.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
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

        # Add skip connection
        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class CactusResNet(nn.Module):
    """
    Shallow Residual Convolutional Network (Custom ResNet) for Cactus identification.
    Structure:
    1. Input Stem (Conv -> BN -> ReLU)
    2. Three Residual Stages (with downsampling in stages 2 and 3)
    3. Global Average Pooling + Linear Classification Head
    """

    def __init__(self, num_classes=1):
        super(CactusResNet, self).__init__()

        # --- Input Stem ---
        # Projects 3-channel input (32x32) to 64-channel feature space
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # --- Residual Stages ---
        # Stage 1: 64 channels, 32x32 output
        self.layer1 = self._make_layer(out_channels=64, stride=1)

        # Stage 2: 128 channels, 16x16 output (downsampled)
        self.layer2 = self._make_layer(out_channels=128, stride=2)

        # Stage 3: 256 channels, 8x8 output (downsampled)
        self.layer3 = self._make_layer(out_channels=256, stride=2)

        # --- Classification Head ---
        # Reduces 8x8x256 feature map to 1x1x256
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear projection to logits
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, stride):
        """
        Helper to create a residual stage.
        """
        layer = ResidualBlock(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def _initialize_weights(self):
        """
        Kaiming initialization for Conv layers and normal init for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Head
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
