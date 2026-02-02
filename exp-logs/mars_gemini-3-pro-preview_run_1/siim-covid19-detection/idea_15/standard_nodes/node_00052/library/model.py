import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DropBlock2D(nn.Module):
    """
    DropBlock regularization layer for Convolutional Neural Networks.
    Drops contiguous regions of feature maps to force the network to learn
    spatially distributed representations.
    """

    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # Only apply during training and if drop_prob > 0
        if not self.training or self.drop_prob <= 0.0:
            return x

        gamma = self._compute_gamma(x)
        mask = self._sample_mask(x, gamma)

        # Normalize the features: x * mask * count / count_kept
        # Add epsilon to denominator to prevent division by zero
        out = x * mask * (mask.numel() / (mask.sum() + 1e-6))
        return out

    def _compute_gamma(self, x):
        """
        Computes gamma (probability of dropping a seed unit) based on the
        desired drop_prob and block_size.
        """
        _, _, H, W = x.shape
        feat_area = H * W
        block_area = self.block_size**2

        valid_H = H - self.block_size + 1
        valid_W = W - self.block_size + 1
        valid_area = valid_H * valid_W

        if valid_area <= 0:
            return 0.0

        gamma = (self.drop_prob * feat_area) / (block_area * valid_area)
        return gamma

    def _sample_mask(self, x, gamma):
        """
        Samples a binary mask where zeros indicate dropped regions.
        """
        N, C, H, W = x.shape

        # 1. Sample drop centers using Bernoulli(gamma)
        # The valid region for centers is (H - block_size + 1, W - block_size + 1)
        if H < self.block_size or W < self.block_size:
            return torch.ones_like(x)

        # Create a mask of random values for the valid region
        p = (
            torch.ones(
                N, C, H - self.block_size + 1, W - self.block_size + 1, device=x.device
            )
            * gamma
        )
        m = torch.bernoulli(p)

        # 2. Pad the mask of centers back to (H, W)
        pad = self.block_size // 2
        m_padded = F.pad(m, (pad, pad, pad, pad), value=0)

        # Handle potential size mismatch due to even/odd block_size
        if m_padded.shape[2] != H or m_padded.shape[3] != W:
            m_padded = F.interpolate(m_padded, size=(H, W), mode="nearest")

        # 3. Expand centers to blocks using MaxPool
        # 1s in m_padded indicate centers of blocks to drop.
        # MaxPool propagates these 1s to the surrounding block_size x block_size area.
        block_mask = F.max_pool2d(
            m_padded,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )

        # Crop to original size if padding caused expansion
        block_mask = block_mask[:, :, :H, :W]

        # 4. Invert mask: We want 1s for kept features, 0s for dropped features
        return 1 - block_mask


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
        x = self.up(x)

        if skip is not None:
            # Handle potential shape mismatch due to rounding in pooling layers
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DropBlockResNet34UNet(nn.Module):
    """
    ResNet34 U-Net with DropBlock Regularization.

    Architecture:
    - Backbone: ResNet34 (Pretrained)
    - Regularization: DropBlock inserted in Layer3 and Layer4
    - Decoder: U-Net style with Bilinear Upsampling
    - Heads:
        1. Classification (GAP + Linear)
        2. Segmentation (Conv)
    """

    def __init__(self):
        super().__init__()

        # Load Pretrained Backbone
        backbone = models.resnet34(pretrained=Config.PRETRAINED)

        # --- Encoder ---
        # Stem (Standard 7x7 Conv)
        self.first_conv = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )  # Out: 64 ch, H/2

        self.maxpool = backbone.maxpool  # Out: H/4

        # Residual Groups
        self.layer1 = backbone.layer1  # Out: 64 ch, H/4
        self.layer2 = backbone.layer2  # Out: 128 ch, H/8
        self.layer3 = backbone.layer3  # Out: 256 ch, H/16
        self.layer4 = backbone.layer4  # Out: 512 ch, H/32

        # Regularization
        self.dropblock = DropBlock2D(
            drop_prob=Config.DROPBLOCK_PROB, block_size=Config.DROPBLOCK_BLOCK_SIZE
        )

        # --- Decoder ---
        # d4: Up(layer4) + layer3 -> 256 ch
        self.dec4 = DecoderBlock(512, 256, 256)
        # d3: Up(dec4) + layer2 -> 128 ch
        self.dec3 = DecoderBlock(256, 128, 128)
        # d2: Up(dec3) + layer1 -> 64 ch
        self.dec2 = DecoderBlock(128, 64, 64)
        # d1: Up(dec2) + first_conv -> 32 ch
        self.dec1 = DecoderBlock(64, 64, 32)

        # --- Heads ---
        # Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # H/2 -> H
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Classification Head (Study Level)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(512, Config.NUM_CLASSES)

    def forward(self, x):
        # --- Encoder Forward ---
        x0 = self.first_conv(x)  # (B, 64, H/2, W/2)
        x_mp = self.maxpool(x0)  # (B, 64, H/4, W/4)

        x1 = self.layer1(x_mp)  # (B, 64, H/4, W/4)
        x2 = self.layer2(x1)  # (B, 128, H/8, W/8)

        x3 = self.layer3(x2)  # (B, 256, H/16, W/16)
        x3 = self.dropblock(x3)  # Apply DropBlock

        x4 = self.layer4(x3)  # (B, 512, H/32, W/32)
        x4 = self.dropblock(x4)  # Apply DropBlock

        # --- Classification Forward ---
        cls_feat = self.global_pool(x4)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_logits = self.cls_head(cls_feat)

        # --- Decoder Forward ---
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        # --- Segmentation Forward ---
        seg_logits = self.seg_head(d1)

        return cls_logits, seg_logits
