import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation (SCSE) Module.
    Combines Channel Squeeze & Excitation (cSE) and Spatial Squeeze & Excitation (sSE).
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()

        # Channel Squeeze and Excitation (cSE)
        # Global Average Pooling -> FC -> ReLU -> FC -> Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation (sSE)
        # Conv 1x1 -> Sigmoid
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        # cSE path: re-weight channels
        # x shape: [B, C, H, W]
        # cse shape: [B, C] -> [B, C, 1, 1]
        cse = self.cSE(x).view(x.size(0), x.size(1), 1, 1)
        x_cse = x * cse

        # sSE path: re-weight spatial locations
        # sse shape: [B, 1, H, W]
        sse = self.sSE(x)
        x_sse = x * sse

        # Combine (Concurrent SCSE)
        return x_cse + x_sse


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with Bilinear Upsampling and SCSE Attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # We use Bilinear upsampling in the forward pass, so no layer here for that.
        # The input to conv block will be in_channels (from upsample) + skip_channels

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

        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate skip connection
        if skip is not None:
            # Handle potential slight shape mismatch due to rounding in pooling
            if x.size() != skip.size():
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

        # Apply Attention
        x = self.scse(x)
        return x


class SoftGatedResnetUNet(nn.Module):
    """
    Soft-Gated Multi-Task ResNet18 U-Net.

    Features:
    - ResNet18 Encoder (pre-trained).
    - 6-Channel Input Adaptation.
    - Parallel Classification Head for Soft Gating.
    - U-Net Decoder with SCSE Attention.
    - Differentiable Soft-Gating: Mask = Seg_Mask * Class_Prob.
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, pretrained=True):
        super(SoftGatedResnetUNet, self).__init__()

        # ===========================
        # Encoder (ResNet18)
        # ===========================
        # Load pre-trained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = models.resnet18(weights=weights)

        # Modify first convolution to accept 6 channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize new input channels
        # We copy the weights from the original 3 channels to the new channels to preserve pre-training info
        with torch.no_grad():
            self.encoder.conv1.weight[:, :3] = original_conv1.weight
            self.encoder.conv1.weight[:, 3:] = original_conv1.weight

        # ===========================
        # Classification Head (Gating)
        # ===========================
        # Attached to Layer 4 (512 channels)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ===========================
        # Decoder
        # ===========================
        # ResNet18 feature channels:
        # x0 (stem): 64
        # x1 (layer1): 64
        # x2 (layer2): 128
        # x3 (layer3): 256
        # x4 (layer4): 512

        # Center block (Bottleneck processing)
        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks
        # Dec4: In 512 (from center), Skip 256 (x3) -> Out 256
        self.dec4 = DecoderBlock(512, 256, 256)

        # Dec3: In 256, Skip 128 (x2) -> Out 128
        self.dec3 = DecoderBlock(256, 128, 128)

        # Dec2: In 128, Skip 64 (x1) -> Out 64
        self.dec2 = DecoderBlock(128, 64, 64)

        # Dec1: In 64, Skip 64 (x0) -> Out 32
        self.dec1 = DecoderBlock(64, 64, 32)

        # Final Upsampling Block (to get to original resolution)
        # Dec1 output is 1/2 resolution (128x128 for 256 input).
        # We need one more upsample.
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # ===========================
        # Encoder Path
        # ===========================
        # Stem
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x0 = self.encoder.relu(x)  # x0: 64ch, 1/2 size

        x = self.encoder.maxpool(x0)

        # Layers
        x1 = self.encoder.layer1(x)  # x1: 64ch, 1/4 size
        x2 = self.encoder.layer2(x1)  # x2: 128ch, 1/8 size
        x3 = self.encoder.layer3(x2)  # x3: 256ch, 1/16 size
        x4 = self.encoder.layer4(x3)  # x4: 512ch, 1/32 size

        # ===========================
        # Classification Head
        # ===========================
        p_cls = self.cls_head(x4)  # Scalar probability [B, 1]

        # ===========================
        # Decoder Path
        # ===========================
        c = self.center(x4)

        d4 = self.dec4(c, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        # Segmentation Mask (Raw)
        m_seg = self.final_conv(d1)

        # ===========================
        # Soft Gating
        # ===========================
        # Multiply pixel-wise mask by global classification probability
        # p_cls is [B, 1], m_seg is [B, 1, H, W]
        # Broadcasting handles the dimensions
        m_final = m_seg * p_cls.view(-1, 1, 1, 1)

        # Return dictionary for SoftGatedLoss
        return {"mask": m_final, "cls": p_cls}


# Alias for backward compatibility
ResnetUNet = SoftGatedResnetUNet
