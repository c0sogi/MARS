import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (SCSE) Module.
    Enhances important features by recalibrating the feature map spatially and channel-wise.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        # Channel Squeeze and Excitation (cSE)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(1, in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // reduction), in_channels, bias=False),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation (sSE)
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Combine channel attention and spatial attention
        # cSE scales channels globally
        cse = self.cSE(x).view(-1, x.size(1), 1, 1) * x
        # sSE scales pixels spatially
        sse = self.sSE(x) * x
        return cse + sse


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module.
    Captures multi-scale context using parallel dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        # 1x1 Convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated Convolution branches
        self.branches = nn.ModuleList(
            [
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
                for rate in rates
            ]
        )

        # Global Average Pooling branch
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection layer to fuse all branches
        # Input channels = out_channels * (1 (1x1) + len(rates) + 1 (pool))
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[-2:]

        # 1x1 branch
        out = [self.conv1x1(x)]

        # Dilated branches
        for b in self.branches:
            out.append(b(x))

        # Global pooling branch (upsampled back to input size)
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        out.append(pooled)

        # Concatenate and project
        return self.project(torch.cat(out, dim=1))


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with SCSE Attention.
    Performs upsampling, concatenation with skip connection, convolution, and attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Convolutional block after concatenation
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

        # Attention mechanism
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle potential rounding errors in dimensions during upsampling
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.scse(x)
        return x


class ContextEnhancedUNet(nn.Module):
    """
    Context-Enhanced ResNet18 U-Net with ASPP and SCSE.

    Encoder: ResNet18 (modified for 6 channels)
    Bridge: ASPP
    Decoder: U-Net style with SCSE attention
    """

    def __init__(self):
        super(ContextEnhancedUNet, self).__init__()

        # ===========================
        # Encoder (ResNet18)
        # ===========================
        # Load pretrained weights
        base_model = models.resnet18(weights="IMAGENET1K_V1")

        # Modify the first layer to accept N_CHANNELS (6) instead of 3
        # We initialize the new weights by copying the pretrained weights
        original_conv1 = base_model.conv1
        self.encoder_conv1 = nn.Conv2d(
            Config.N_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize weights: Copy RGB weights to first 3 channels, and again to next 3
        # This preserves the pretrained feature extraction capabilities for the Ash channels
        with torch.no_grad():
            self.encoder_conv1.weight[:, :3] = original_conv1.weight
            self.encoder_conv1.weight[:, 3:] = original_conv1.weight

        self.encoder_bn1 = base_model.bn1
        self.encoder_relu = base_model.relu
        self.encoder_maxpool = base_model.maxpool

        self.encoder_layer1 = base_model.layer1  # 64 channels
        self.encoder_layer2 = base_model.layer2  # 128 channels
        self.encoder_layer3 = base_model.layer3  # 256 channels
        self.encoder_layer4 = base_model.layer4  # 512 channels

        # ===========================
        # Bridge (ASPP)
        # ===========================
        # Input: 512 (Layer4), Output: 256
        self.aspp = ASPP(in_channels=512, out_channels=256)

        # ===========================
        # Decoder
        # ===========================
        # Config.DECODER_CHANNELS = (256, 128, 64, 32, 16)

        # Block 1: Input from ASPP (256), Skip from Layer3 (256) -> Out 256
        self.decoder_block1 = DecoderBlock(
            in_channels=256, skip_channels=256, out_channels=256
        )

        # Block 2: Input from Block1 (256), Skip from Layer2 (128) -> Out 128
        self.decoder_block2 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Block 3: Input from Block2 (128), Skip from Layer1 (64) -> Out 64
        self.decoder_block3 = DecoderBlock(
            in_channels=128, skip_channels=64, out_channels=64
        )

        # Block 4: Input from Block3 (64), Skip from Relu/Layer0 (64) -> Out 32
        self.decoder_block4 = DecoderBlock(
            in_channels=64, skip_channels=64, out_channels=32
        )

        # Final Upsampling Block (to original resolution)
        # Input from Block4 (32), Upsample to 256x256 -> Out 16
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Classification Head
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Input: (B, 6, H, W)
        x0 = self.encoder_conv1(x)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)  # (B, 64, H/2, W/2)

        x_pool = self.encoder_maxpool(x0)  # (B, 64, H/4, W/4)

        x1 = self.encoder_layer1(x_pool)  # (B, 64, H/4, W/4)
        x2 = self.encoder_layer2(x1)  # (B, 128, H/8, W/8)
        x3 = self.encoder_layer3(x2)  # (B, 256, H/16, W/16)
        x4 = self.encoder_layer4(x3)  # (B, 512, H/32, W/32)

        # --- Bridge ---
        x_bridge = self.aspp(x4)  # (B, 256, H/32, W/32)

        # --- Decoder ---
        d1 = self.decoder_block1(x_bridge, x3)  # -> (B, 256, H/16, W/16)
        d2 = self.decoder_block2(d1, x2)  # -> (B, 128, H/8, W/8)
        d3 = self.decoder_block3(d2, x1)  # -> (B, 64, H/4, W/4)
        d4 = self.decoder_block4(d3, x0)  # -> (B, 32, H/2, W/2)

        # --- Head ---
        out = self.final_upsample(d4)  # -> (B, 16, H, W)
        logits = self.head(out)  # -> (B, 1, H, W)

        return logits
