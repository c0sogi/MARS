import torch
import torch.nn as nn
from torchvision import models
import random
import numpy as np


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Set seed immediately upon import
set_seed(42)


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block.
    Performs: 1x1 Conv (reduce) -> 3x3 Transposed Conv (upsample) -> 1x1 Conv (expand).
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Internal width is typically out_channels // 4 in standard LinkNet
        internal_channels = out_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv to reduce channels
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv to upsample
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv to expand channels
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthRegressor(nn.Module):
    """
    ResNet18-based model to regress depth from seismic images.
    Used to impute depths for the test set.
    """

    def __init__(self):
        super(DepthRegressor, self).__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)

        # Modify first layer for 1-channel input (Grayscale)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize the new conv1 weights by averaging original RGB weights
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv1.weight.data.mean(
                dim=1, keepdim=True
            )

        # Modify the fully connected layer for scalar regression
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.backbone(x)


class DepthAwareLinkNet34(nn.Module):
    """
    ResNet34-LinkNet with Depth Injection.
    """

    def __init__(self, num_classes=1):
        super(DepthAwareLinkNet34, self).__init__()

        # ------------------------------------------------------------------
        # Encoder: ResNet34
        # ------------------------------------------------------------------
        resnet = models.resnet34(pretrained=True)

        # Modify first layer for 1-channel input
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.mean(
                dim=1, keepdim=True
            )

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # Output: 64 ch, 1/4 res
        self.layer2 = resnet.layer2  # Output: 128 ch, 1/8 res
        self.layer3 = resnet.layer3  # Output: 256 ch, 1/16 res
        self.layer4 = resnet.layer4  # Output: 512 ch, 1/32 res

        # ------------------------------------------------------------------
        # Depth Injection
        # ------------------------------------------------------------------
        self.depth_embedding = nn.Sequential(nn.Linear(1, 16), nn.ReLU(inplace=True))

        # ------------------------------------------------------------------
        # Decoder: LinkNet style
        # ------------------------------------------------------------------
        # Decoder 4: Takes Bottleneck (512 + 16 depth) -> Output 256
        self.decoder4 = DecoderBlock(512 + 16, 256)

        # Decoder 3: Takes 256 -> Output 128
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: Takes 128 -> Output 64
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: Takes 64 -> Output 64
        self.decoder1 = DecoderBlock(64, 64)

        # ------------------------------------------------------------------
        # Final Head
        # ------------------------------------------------------------------
        # Upsample from 1/2 resolution (output of decoder1) to 1/1 resolution
        self.final_head = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x, depth):
        # ------------------------------------------------------------------
        # Encoder Pass
        # ------------------------------------------------------------------
        x0 = self.conv1(x)  # 1/2
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)  # 1/4

        e1 = self.layer1(x1)  # 1/4, 64
        e2 = self.layer2(e1)  # 1/8, 128
        e3 = self.layer3(e2)  # 1/16, 256
        e4 = self.layer4(e3)  # 1/32, 512

        # ------------------------------------------------------------------
        # Depth Injection at Bottleneck
        # ------------------------------------------------------------------
        d = self.depth_embedding(depth)  # (B, 16)
        d = d.unsqueeze(2).unsqueeze(3)  # (B, 16, 1, 1)
        d = d.expand(-1, -1, e4.size(2), e4.size(3))  # (B, 16, H/32, W/32)

        bottleneck = torch.cat([e4, d], dim=1)  # (B, 528, H/32, W/32)

        # ------------------------------------------------------------------
        # Decoder Pass with Additive Skip Connections
        # ------------------------------------------------------------------
        # Dec 4
        d4 = self.decoder4(bottleneck)  # -> 256, 1/16
        d4 = d4 + e3

        # Dec 3
        d3 = self.decoder3(d4)  # -> 128, 1/8
        d3 = d3 + e2

        # Dec 2
        d2 = self.decoder2(d3)  # -> 64, 1/4
        d2 = d2 + e1

        # Dec 1
        d1 = self.decoder1(d2)  # -> 64, 1/2
        # Skip connection: x0 is the feature map before maxpool (stride 2, 64 channels)
        d1 = d1 + x0

        # ------------------------------------------------------------------
        # Final Prediction
        # ------------------------------------------------------------------
        out = self.final_head(d1)  # -> 1, 1/1

        return out
