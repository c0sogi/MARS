import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concatenate -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Input channels = upsampled channels + skip connection channels
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
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Handle skip connection
        if skip is not None:
            # Handle potential padding issues (though 512x512 is standard)
            if x.shape[2:] != skip.shape[2:]:
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


class ResNet18UNetMultiScale(nn.Module):
    """
    ResNet18 U-Net with Multi-Scale Aggregated Classification Head.

    Encoder: ResNet18 (ImageNet weights)
    Decoder: U-Net style with symmetric skips
    Cls Head: Aggregates Layer 2, 3, 4 via GAP -> Concat -> Linear
    Seg Head: 1x1 Conv
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(ResNet18UNetMultiScale, self).__init__()

        # 1. Encoder (ResNet18)
        # Load weights if requested
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.encoder = models.resnet18(weights=weights)
        else:
            self.encoder = models.resnet18(weights=None)

        # Remove fully connected layer and avgpool to use as feature extractor
        # We will access layers directly in forward pass

        # Encoder Channels:
        # x0 (conv1): 64
        # layer1: 64
        # layer2: 128
        # layer3: 256
        # layer4: 512

        # 2. Multi-Scale Classification Head
        # Inputs: Layer 2 (128), Layer 3 (256), Layer 4 (512)
        # Total concatenated features: 128 + 256 + 512 = 896
        self.cls_head = nn.Linear(128 + 256 + 512, num_classes)

        # 3. Decoder
        # Bottleneck is Layer 4 (512)

        # Block 4: Up(Layer4) + Layer3 -> 512 + 256 -> 256
        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )

        # Block 3: Up(Block4) + Layer2 -> 256 + 128 -> 128
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Block 2: Up(Block3) + Layer1 -> 128 + 64 -> 64
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Block 1: Up(Block2) + x0 (conv1) -> 64 + 64 -> 32
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final Block: Up(Block1) -> 32 + 0 -> 16 (To restore full resolution)
        self.decoder0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 4. Segmentation Head
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Input: (B, 3, H, W)

        # Stem
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x0 = self.encoder.relu(x)  # (B, 64, H/2, W/2)

        x = self.encoder.maxpool(x0)  # (B, 64, H/4, W/4)

        # Layers
        x1 = self.encoder.layer1(x)  # (B, 64, H/4, W/4)
        x2 = self.encoder.layer2(x1)  # (B, 128, H/8, W/8)
        x3 = self.encoder.layer3(x2)  # (B, 256, H/16, W/16)
        x4 = self.encoder.layer4(x3)  # (B, 512, H/32, W/32)

        # --- Classification Head (Multi-Scale) ---
        # GAP on Layer 2, 3, 4
        p2 = F.adaptive_avg_pool2d(x2, (1, 1)).flatten(1)  # (B, 128)
        p3 = F.adaptive_avg_pool2d(x3, (1, 1)).flatten(1)  # (B, 256)
        p4 = F.adaptive_avg_pool2d(x4, (1, 1)).flatten(1)  # (B, 512)

        # Concatenate
        cls_feat = torch.cat([p2, p3, p4], dim=1)  # (B, 896)

        # Predict
        logit_cls = self.cls_head(cls_feat)  # (B, 4)

        # --- Decoder ---
        d4 = self.decoder4(x4, x3)  # -> (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, x2)  # -> (B, 128, H/8, W/8)
        d2 = self.decoder2(d3, x1)  # -> (B, 64, H/4, W/4)
        d1 = self.decoder1(d2, x0)  # -> (B, 32, H/2, W/2)
        d0 = self.decoder0(d1)  # -> (B, 16, H, W)

        # --- Segmentation Head ---
        logit_mask = self.seg_head(d0)  # (B, 1, H, W)

        return logit_mask, logit_cls
