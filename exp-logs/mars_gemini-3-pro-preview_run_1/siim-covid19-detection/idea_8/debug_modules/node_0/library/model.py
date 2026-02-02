import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class ASPPConv(nn.Sequential):
    """
    Helper class for ASPP 3x3 convolution with dilation.
    """

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
    Helper class for ASPP global pooling branch.
    """

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module.
    """

    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 Conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous Convs
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global Pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Project to single output channel dimension
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> Conv -> Conv.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
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

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate with skip connection
        if skip is not None:
            # Handle potential rounding errors in dimensions
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet18UNetASPP(nn.Module):
    """
    ResNet18 U-Net with ASPP bottleneck and Multi-Task Heads.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(ResNet18UNetASPP, self).__init__()

        # --- Encoder (ResNet18) ---
        # Using IMAGENET1K_V1 weights
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.base_model = models.resnet18(weights=weights)

        self.encoder_layers = list(self.base_model.children())

        # Split encoder into stages for skip connections
        # Input: (B, 3, H, W)
        self.layer0 = nn.Sequential(
            *self.encoder_layers[:3]
        )  # Conv1+BN+ReLU -> (64, H/2, W/2)
        self.maxpool = self.encoder_layers[3]  # MaxPool      -> (64, H/4, W/4)
        self.layer1 = self.encoder_layers[4]  # Layer1       -> (64, H/4, W/4)
        self.layer2 = self.encoder_layers[5]  # Layer2       -> (128, H/8, W/8)
        self.layer3 = self.encoder_layers[6]  # Layer3       -> (256, H/16, W/16)
        self.layer4 = self.encoder_layers[7]  # Layer4       -> (512, H/32, W/32)

        # --- Bottleneck (ASPP) ---
        # Input: 512 channels (from layer4) -> Output: 256 channels
        self.aspp = ASPP(in_channels=512, out_channels=256, atrous_rates=[6, 12, 18])

        # --- Classification Head ---
        # Attached to ASPP output
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        # --- Decoder ---
        # Block 4: Up from ASPP (256) + Skip Layer 3 (256) -> Out 256
        self.decoder4 = DecoderBlock(
            in_channels=256, skip_channels=256, out_channels=256
        )

        # Block 3: Up from Dec4 (256) + Skip Layer 2 (128) -> Out 128
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Block 2: Up from Dec3 (128) + Skip Layer 1 (64) -> Out 64
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Block 1: Up from Dec2 (64) + Skip Layer 0 (64) -> Out 32
        # Note: Layer 0 is before MaxPool, so it is H/2 resolution
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # --- Segmentation Head ---
        # Final Upsample to original resolution (H, W)
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary output
        )

    def forward(self, x):
        # --- Encoder ---
        x0 = self.layer0(x)  # Stride 2
        x_pool = self.maxpool(x0)  # Stride 4
        x1 = self.layer1(x_pool)  # Stride 4
        x2 = self.layer2(x1)  # Stride 8
        x3 = self.layer3(x2)  # Stride 16
        x4 = self.layer4(x3)  # Stride 32

        # --- Bottleneck ---
        x_aspp = self.aspp(x4)  # Stride 32, 256 ch

        # --- Classification ---
        x_cls = self.avgpool(x_aspp)
        x_cls = torch.flatten(x_cls, 1)
        logits_cls = self.fc(x_cls)

        # --- Decoder ---
        d4 = self.decoder4(x_aspp, x3)  # Up to Stride 16
        d3 = self.decoder3(d4, x2)  # Up to Stride 8
        d2 = self.decoder2(d3, x1)  # Up to Stride 4
        d1 = self.decoder1(d2, x0)  # Up to Stride 2

        # --- Segmentation ---
        logits_seg = self.final_conv(d1)  # Up to Stride 1

        return logits_cls, logits_seg
