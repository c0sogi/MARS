import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import MODEL_CONFIG


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Module.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    """

    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super().__init__()
        modules = []
        # 1x1 conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous convolutions
        for rate in rates[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global Average Pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)

        # Project after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))

        # Handle global pooling separately to upsample back to input size
        gap = self.convs[-1](x)
        gap = F.interpolate(gap, size=x.shape[2:], mode="bilinear", align_corners=False)
        res.append(gap)

        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Decoder block with Bilinear Upsampling, Concatenation, Convolutions, and SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels, out_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Ensure dimensions match (handle potential rounding errors in odd sizes, though unlikely here)
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.scse(x)
        return x


class DilatedResNetUNet(nn.Module):
    """
    Dilated ResNet18 U-Net with Output Stride 8.
    """

    def __init__(self, config=MODEL_CONFIG):
        super().__init__()
        in_channels = config.get("in_channels", 6)
        classes = config.get("classes", 1)

        # Encoder: ResNet18 with Output Stride = 8
        # Standard ResNet18 (BasicBlock) does not support replace_stride_with_dilation in torchvision.
        # We manually adjust strides and dilations for Layer 3 and Layer 4 to achieve Output Stride 8.
        self.encoder = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        # Layer 3: Stride 2 -> 1, Dilation 1 -> 2
        for m in self.encoder.layer3.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (2, 2)
                    m.padding = (2, 2)
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
        # Adjust downsample layer in the first block of Layer 3
        if self.encoder.layer3[0].downsample is not None:
            self.encoder.layer3[0].downsample[0].stride = (1, 1)

        # Layer 4: Stride 2 -> 1, Dilation 1 -> 4
        for m in self.encoder.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (4, 4)
                    m.padding = (4, 4)
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
        # Adjust downsample layer in the first block of Layer 4
        if self.encoder.layer4[0].downsample is not None:
            self.encoder.layer4[0].downsample[0].stride = (1, 1)

        # Modify first convolution to accept N_CHANNELS
        original_conv1 = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        # Initialize the new conv1 weights (Kaiming Normal)
        nn.init.kaiming_normal_(
            self.encoder.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

        # Encoder Channel Sizes:
        # Layer 4: 512
        # Layer 1: 64
        # Conv1 (Relu): 64

        # Bottleneck: ASPP
        # Input: 512 (Layer 4), Output: 256
        self.aspp = ASPP(512, 256)

        # Decoder
        # Block 1: 32x32 -> 64x64
        # Input: 256 (ASPP). Skip: Layer1 (64). Output: 128.
        self.dec1 = DecoderBlock(256, 64, 128)

        # Block 2: 64x64 -> 128x128
        # Input: 128 (Dec1). Skip: Conv1_Relu (64). Output: 64.
        self.dec2 = DecoderBlock(128, 64, 64)

        # Block 3: 128x128 -> 256x256
        # Input: 64 (Dec2). Skip: None. Output: 32.
        self.dec3 = DecoderBlock(64, 0, 32)

        # Final Head
        self.head = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Input: (B, C, 256, 256)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x0 = self.encoder.relu(x)  # (B, 64, 128, 128) -> Skip for Dec2

        x = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x)  # (B, 64, 64, 64)   -> Skip for Dec1
        x = self.encoder.layer2(x1)  # (B, 128, 32, 32)
        x = self.encoder.layer3(x)  # (B, 256, 32, 32)  -> Dilated
        x = self.encoder.layer4(x)  # (B, 512, 32, 32)  -> Dilated

        # --- Bottleneck ---
        x = self.aspp(x)  # (B, 256, 32, 32)

        # --- Decoder ---
        x = self.dec1(x, x1)  # (B, 128, 64, 64)
        x = self.dec2(x, x0)  # (B, 64, 128, 128)
        x = self.dec3(x)  # (B, 32, 256, 256)

        # --- Head ---
        logits = self.head(x)  # (B, 1, 256, 256)

        return logits
