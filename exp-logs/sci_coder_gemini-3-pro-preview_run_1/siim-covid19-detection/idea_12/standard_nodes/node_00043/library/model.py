import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the p-norm of the input feature map.
    p is a learnable parameter.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp min value to avoid NaN gradients
        x = x.clamp(min=eps)
        # Average pooling on x^p
        x = x.pow(p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # Raise to 1/p
        x = x.pow(1.0 / p)
        return x


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block.
    Performs Bilinear Upsampling -> Concatenation with Skip -> ConvBlock.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

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
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Handle Skip Connection
        if skip is not None:
            # Ensure spatial dimensions match (handle slight rounding diffs if any)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNet18D_UNet(nn.Module):
    """
    Multi-Task U-Net with ResNet18-D Backbone and GeM Pooling.
    """

    def __init__(self, num_classes=Config.num_study_classes, pretrained=True):
        super(ResNet18D_UNet, self).__init__()

        # 1. Encoder (ResNet18-D)
        # features_only=True returns a list of feature maps
        # ResNet18d strides: [2, 4, 8, 16, 32]
        # Channels: [64, 64, 128, 256, 512]
        self.encoder = timm.create_model(
            Config.backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Encoder channels
        enc_channels = [64, 64, 128, 256, 512]

        # 2. Classification Head (Study Level)
        # Attached to the bottleneck (deepest feature map)
        self.gem_pool = GeM(p=3.0)
        self.cls_head = nn.Linear(enc_channels[-1], num_classes)

        # 3. Decoder (Segmentation Level)
        # We decode from 512 down to original resolution

        # Block 4: 512 (x) + 256 (skip) -> 256
        self.dec4 = DecoderBlock(enc_channels[4], enc_channels[3], 256)

        # Block 3: 256 (x) + 128 (skip) -> 128
        self.dec3 = DecoderBlock(256, enc_channels[2], 128)

        # Block 2: 128 (x) + 64 (skip) -> 64
        self.dec2 = DecoderBlock(128, enc_channels[1], 64)

        # Block 1: 64 (x) + 64 (skip) -> 64
        self.dec1 = DecoderBlock(64, enc_channels[0], 64)

        # Final Upsample: 64 (x) -> 32 (No skip from raw image usually in this design)
        # We upsample one last time to reach stride 1 (original size)
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Output 1 channel for binary mask
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, 3, H, W)
        Returns:
            logits (torch.Tensor): Study-level classification logits (B, 4)
            mask (torch.Tensor): Image-level segmentation mask logits (B, 1, H, W)
        """
        # --- Encoder ---
        # features is a list of tensors [c1, c2, c3, c4, c5]
        # c1: stride 2, c2: stride 4, ..., c5: stride 32
        features = self.encoder(x)

        c1, c2, c3, c4, c5 = features

        # --- Classification Path ---
        # Use the deepest feature map (c5)
        x_cls = self.gem_pool(c5)  # (B, 512, 1, 1)
        x_cls = torch.flatten(x_cls, 1)  # (B, 512)
        logits = self.cls_head(x_cls)  # (B, 4)

        # --- Segmentation Path ---
        # U-Net Decoder with Skips
        d4 = self.dec4(c5, c4)  # Stride 32 -> 16
        d3 = self.dec3(d4, c3)  # Stride 16 -> 8
        d2 = self.dec2(d3, c2)  # Stride 8 -> 4
        d1 = self.dec1(d2, c1)  # Stride 4 -> 2

        # Final upsample to Stride 1
        mask = self.final_conv(d1)  # Stride 2 -> 1

        return logits, mask
