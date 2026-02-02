import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # We concatenate the upsampled input with the skip connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # Upsample x to match skip's spatial resolution
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ResNet18_UNet(nn.Module):
    """
    Standard ResNet18 U-Net with Global Average Pooling.
    Optimized based on Lesson 00046: Simpler backbone and pooling often outperform
    complex variants (GeM, ResNet-D) for diffuse medical opacities.

    Backbone: resnet18
    Heads:
      1. Classification: GAP -> Linear (Study Level)
      2. Segmentation: U-Net Decoder -> Conv 1x1 (Image Level)
    """

    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet18_UNet, self).__init__()

        # 1. Backbone: Standard ResNet18
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get channel counts: [64, 64, 128, 256, 512]
        encoder_channels = self.encoder.feature_info.channels()

        # 2. Classification Head (Study Level)
        # Use Standard GAP (Cite solution_lesson_node_00046)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(encoder_channels[-1], num_classes)

        # 3. Segmentation Decoder (Image Level)
        # We build decoder blocks moving from deep to shallow
        # e4 (512) -> e3 (256) -> e2 (128) -> e1 (64) -> e0 (64)

        # Block 4: Input e4 (512), Skip e3 (256) -> Out 256
        self.dec4 = DecoderBlock(encoder_channels[4], encoder_channels[3], 256)

        # Block 3: Input dec4 (256), Skip e2 (128) -> Out 128
        self.dec3 = DecoderBlock(256, encoder_channels[2], 128)

        # Block 2: Input dec3 (128), Skip e1 (64) -> Out 64
        self.dec2 = DecoderBlock(128, encoder_channels[1], 64)

        # Block 1: Input dec2 (64), Skip e0 (64) -> Out 32
        self.dec1 = DecoderBlock(64, encoder_channels[0], 32)

        # Final Upsample to original resolution
        # Input dec1 (32) -> Upsample -> Conv -> 1 channel
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, 3, H, W)
        Returns:
            cls_logits: (B, num_classes)
            seg_logits: (B, 1, H, W)
        """
        input_shape = x.shape[-2:]

        # --- Encoder ---
        # Features: [e0, e1, e2, e3, e4]
        # Strides:  [2,  4,  8,  16, 32]
        features = self.encoder(x)

        e0, e1, e2, e3, e4 = features

        # --- Classification Head ---
        # Apply GeM on the deepest feature map
        x_cls = self.global_pool(e4)
        x_cls = x_cls.flatten(1)
        cls_logits = self.cls_head(x_cls)

        # --- Segmentation Decoder ---
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Final upsample to match input resolution
        # d1 is stride 2 (half resolution) relative to input
        d_final = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)

        seg_logits = self.final_conv(d_final)

        return cls_logits, seg_logits
