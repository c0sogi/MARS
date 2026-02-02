import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (SCSE) Block.
    Ref: Roy et al., "Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks"
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()

        # Channel Squeeze and Excitation (cSE)
        # Squeeze: Global Average Pooling
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation (sSE)
        # Squeeze: 1x1 Conv
        # Excitation: Sigmoid
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        # cSE path: re-weight channels
        chn_se = self.cSE(x) * x

        # sSE path: re-weight spatial locations
        spa_se = self.sSE(x) * x

        # Concurrent combination
        return chn_se + spa_se


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with SCSE Attention.
    Performs: Upsample -> Concat -> SCSE -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Total channels after concatenation
        total_in_channels = in_channels + skip_channels

        # SCSE Block applied after concatenation to suppress noise from skips
        self.attention = SCSEBlock(total_in_channels)

        # Standard Double Conv
        self.conv = nn.Sequential(
            nn.Conv2d(
                total_in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate with skip connection if provided
        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        # Apply Attention
        x = self.attention(x)

        # Refine features
        x = self.conv(x)
        return x


class MultiTaskResNetUNet(nn.Module):
    """
    Multi-Task U-Net with ResNet18 Encoder and SCSE Attention Decoder.

    Tasks:
    1. Segmentation: Pixel-wise binary mask.
    2. Classification: Image-level binary label (Contrail vs No Contrail).
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, pretrained=True):
        super(MultiTaskResNetUNet, self).__init__()

        # ------------------------------------------------------------------
        # Encoder: ResNet18
        # ------------------------------------------------------------------
        weights = "IMAGENET1K_V1" if pretrained else None
        self.encoder = torchvision.models.resnet18(weights=weights)

        # Modify first conv layer to accept 'in_channels' (e.g., 6)
        # We copy the weights of the first 3 channels to the new channels to preserve initialization
        original_conv1 = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        if pretrained:
            with torch.no_grad():
                # Copy RGB weights to the first 3 channels
                self.encoder.conv1.weight[:, :3] = original_conv1.weight
                # Copy RGB weights to the next 3 channels (temporal diff) as a reasonable init
                if in_channels > 3:
                    self.encoder.conv1.weight[:, 3:6] = original_conv1.weight

        # ------------------------------------------------------------------
        # Classification Head (Attached to Bottleneck)
        # ------------------------------------------------------------------
        # ResNet18 Layer 4 output channels: 512
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),  # Output logit
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # ResNet18 Feature Dimensions (assuming 256x256 input):
        # Layer 4: (512, 8, 8)
        # Layer 3: (256, 16, 16)
        # Layer 2: (128, 32, 32)
        # Layer 1: (64, 64, 64)
        # Conv1 (Pre-Pool): (64, 128, 128)

        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final Upsample to restore 256x256
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Segmentation Head
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # ------------------------------------------------------------------
        # Encoder Forward
        # ------------------------------------------------------------------
        # Input: (B, 6, 256, 256)
        x0 = self.encoder.conv1(x)  # (B, 64, 128, 128)
        x0_bn = self.encoder.bn1(x0)
        x0_relu = self.encoder.relu(x0_bn)

        x_pool = self.encoder.maxpool(x0_relu)  # (B, 64, 64, 64)

        x1 = self.encoder.layer1(x_pool)  # (B, 64, 64, 64)
        x2 = self.encoder.layer2(x1)  # (B, 128, 32, 32)
        x3 = self.encoder.layer3(x2)  # (B, 256, 16, 16)
        x4 = self.encoder.layer4(x3)  # (B, 512, 8, 8)

        # ------------------------------------------------------------------
        # Classification Task
        # ------------------------------------------------------------------
        cls_logits = self.cls_head(x4)

        # ------------------------------------------------------------------
        # Decoder Forward
        # ------------------------------------------------------------------
        d4 = self.decoder4(x4, x3)  # (B, 256, 16, 16)
        d3 = self.decoder3(d4, x2)  # (B, 128, 32, 32)
        d2 = self.decoder2(d3, x1)  # (B, 64, 64, 64)

        # Note: ResNet18 doesn't have a skip connection at 128x128 in layer structure,
        # but we can use the output of conv1 (x0_relu) which is 128x128.
        d1 = self.decoder1(d2, x0_relu)  # (B, 32, 128, 128)

        # Final upsample to 256x256
        out = self.final_upsample(d1)  # (B, 16, 256, 256)

        # Segmentation Logits
        seg_logits = self.seg_head(out)  # (B, 1, 256, 256)

        return seg_logits, cls_logits


ResNetUNet = MultiTaskResNetUNet
