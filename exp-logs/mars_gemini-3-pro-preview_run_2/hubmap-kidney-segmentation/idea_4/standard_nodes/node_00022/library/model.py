import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class ConvBlock(nn.Module):
    """
    Basic Convolutional Block: Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiTaskResNetFPN(nn.Module):
    """
    Multi-Task FPN with ResNet-34 Backbone.

    Architecture:
    1. Encoder: ResNet-34 (pretrained)
    2. Neck: FPN (Feature Pyramid Network)
    3. Decoder: Semantic Segmentation Aggregation
    4. Heads:
       - Primary: Glomerulus Segmentation
       - Auxiliary: Anatomical Structure (Cortex) Segmentation
    """

    def __init__(self):
        super(MultiTaskResNetFPN, self).__init__()

        # ==========================================
        # 1. Backbone (ResNet-34)
        # ==========================================
        # Load pretrained weights
        weights = (
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            if Config.ENCODER_WEIGHTS == "imagenet"
            else None
        )
        self.backbone = torchvision.models.resnet34(weights=weights)

        # Remove fully connected layer and avgpool as we only need features
        # We will access layers explicitly in forward()

        # Channel counts for ResNet34 layers:
        # layer1 (C2): 64
        # layer2 (C3): 128
        # layer3 (C4): 256
        # layer4 (C5): 512

        # ==========================================
        # 2. FPN Neck
        # ==========================================
        self.fpn_out_channels = 256

        # Lateral layers (1x1 conv to adjust channels)
        self.lat_layer1 = nn.Conv2d(64, self.fpn_out_channels, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(128, self.fpn_out_channels, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(256, self.fpn_out_channels, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(512, self.fpn_out_channels, kernel_size=1)

        # Smooth layers (3x3 conv to reduce aliasing)
        self.smooth_layer1 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1
        )
        self.smooth_layer2 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1
        )
        self.smooth_layer3 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1
        )
        self.smooth_layer4 = nn.Conv2d(
            self.fpn_out_channels, self.fpn_out_channels, kernel_size=3, padding=1
        )

        # ==========================================
        # 3. Aggregation Block
        # ==========================================
        # We concatenate P2, P3, P4, P5 (all upsampled to P2 size)
        # Input channels = 4 * 256 = 1024
        self.aggregation = nn.Sequential(
            ConvBlock(self.fpn_out_channels * 4, 256, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
        )

        # ==========================================
        # 4. Multi-Task Heads
        # ==========================================
        # Primary Head: Glomerulus (Binary)
        self.head_glom = nn.Sequential(
            ConvBlock(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        # Auxiliary Head: Cortex (Binary)
        self.head_cortex = nn.Sequential(
            ConvBlock(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def _upsample_add(self, x, y):
        """Upsample x and add it to y."""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self, x):
        # ==========================================
        # Encoder Forward
        # ==========================================
        # Initial block
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # ResNet Layers
        c2 = self.backbone.layer1(x)  # stride 4
        c3 = self.backbone.layer2(c2)  # stride 8
        c4 = self.backbone.layer3(c3)  # stride 16
        c5 = self.backbone.layer4(c4)  # stride 32

        # ==========================================
        # FPN Forward
        # ==========================================
        # Top-down pathway
        p5 = self.lat_layer4(c5)
        p4 = self._upsample_add(p5, self.lat_layer3(c4))
        p3 = self._upsample_add(p4, self.lat_layer2(c3))
        p2 = self._upsample_add(p3, self.lat_layer1(c2))

        # Smooth
        p5 = self.smooth_layer4(p5)
        p4 = self.smooth_layer3(p4)
        p3 = self.smooth_layer2(p3)
        p2 = self.smooth_layer1(p2)

        # ==========================================
        # Aggregation
        # ==========================================
        # Upsample all to P2 resolution (1/4 input size)
        target_h, target_w = p2.size(2), p2.size(3)

        p5_up = F.interpolate(
            p5, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        p4_up = F.interpolate(
            p4, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        p3_up = F.interpolate(
            p3, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        p2_up = p2  # Already at target size

        # Concatenate
        features = torch.cat([p2_up, p3_up, p4_up, p5_up], dim=1)
        features = self.aggregation(features)

        # ==========================================
        # Heads
        # ==========================================
        out_glom = self.head_glom(features)
        out_cortex = self.head_cortex(features)

        # Concatenate outputs: Channel 0 = Glom, Channel 1 = Cortex
        out = torch.cat([out_glom, out_cortex], dim=1)

        # Final Upsample to input resolution (4x)
        # Note: Input was downsampled by 4 to get to C2/P2
        # We rely on interpolation to restore full size
        # Assuming input dimensions are divisible by 32, this works cleanly.
        # If not, we might need to reference original input size.
        # However, F.interpolate with scale_factor=4 is standard here.
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)

        return out
