import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Block.
    Concurrent Spatial and Channel attention.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        # Channel Squeeze and Excitation
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # x * cSE + x * sSE
        return x * self.cSE(x) + x * self.sSE(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    """

    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[0],
                dilation=rates[0],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[1],
                dilation=rates[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=rates[2],
                dilation=rates[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)

        # Global pooling requires resizing back to input spatial dimensions
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode="bilinear", align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.project(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with SCSE Attention.
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

        self.attention = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.size()[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x


class ResNetUNet(nn.Module):
    """
    ResNet18-based U-Net with ASPP and SCSE.
    """

    def __init__(self, in_channels, out_classes=1, backbone_weights="IMAGENET1K_V1"):
        super(ResNetUNet, self).__init__()

        # Load Pretrained ResNet18
        # We use weights=None if we want to initialize from scratch, but prompt implies using pretrained capacity
        # We will handle the input channel mismatch below.
        if backbone_weights == "imagenet":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.encoder = models.resnet18(weights=weights)

        # Modify first convolution layer to accept 'in_channels'
        # ResNet18 conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize new conv1 weights
        # Copy weights for the first 3 channels, average for the rest
        with torch.no_grad():
            if original_conv1.weight.shape[1] == 3:
                self.encoder.conv1.weight[:, :3] = original_conv1.weight
                # Initialize remaining channels with average of original weights
                if in_channels > 3:
                    avg_weight = torch.mean(original_conv1.weight, dim=1, keepdim=True)
                    for i in range(3, in_channels):
                        self.encoder.conv1.weight[:, i : i + 1] = avg_weight
            else:
                nn.init.kaiming_normal_(
                    self.encoder.conv1.weight, mode="fan_out", nonlinearity="relu"
                )

        # Encoder Layers
        # layer0: conv1 -> bn1 -> relu (before maxpool)
        self.layer0 = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu
        )
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1  # 64 channels
        self.layer2 = self.encoder.layer2  # 128 channels
        self.layer3 = self.encoder.layer3  # 256 channels
        self.layer4 = self.encoder.layer4  # 512 channels

        # Bottleneck
        self.aspp = ASPP(512, 256)

        # Decoder
        # Layer 4 (512) -> ASPP (256)
        # Decoder 4: Up(256) + Layer 3 (256) -> 256
        self.decoder4 = DecoderBlock(256, 256, 256)

        # Decoder 3: Up(256) + Layer 2 (128) -> 128
        self.decoder3 = DecoderBlock(256, 128, 128)

        # Decoder 2: Up(128) + Layer 1 (64) -> 64
        self.decoder2 = DecoderBlock(128, 64, 64)

        # Decoder 1: Up(64) + Layer 0 (64) -> 64
        self.decoder1 = DecoderBlock(64, 64, 64)

        # Final Upsample to original resolution (no skip from raw input usually in this design)
        # We just project to classes. Since decoder1 output is H/2, W/2, we need one more upsample.
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_classes, 1),
        )

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)  # H/2, 64
        x_pool = self.maxpool(x0)  # H/4, 64

        x1 = self.layer1(x_pool)  # H/4, 64
        x2 = self.layer2(x1)  # H/8, 128
        x3 = self.layer3(x2)  # H/16, 256
        x4 = self.layer4(x3)  # H/32, 512

        # Bottleneck
        x_mid = self.aspp(x4)  # H/32, 256

        # Decoder
        d4 = self.decoder4(x_mid, x3)  # H/16, 256
        d3 = self.decoder3(d4, x2)  # H/8, 128
        d2 = self.decoder2(d3, x1)  # H/4, 64
        d1 = self.decoder1(d2, x0)  # H/2, 64

        out = self.final_conv(d1)  # H, 1

        return out


class CascadedUNet(nn.Module):
    """
    Two-stage Cascaded U-Net.
    Stage 1: Detection (High Recall)
    Stage 2: Refinement (Shape Delineation)
    """

    def __init__(self):
        super(CascadedUNet, self).__init__()

        # Stage 1: Takes original input (6 channels)
        self.stage1 = ResNetUNet(
            in_channels=Config.IN_CHANNELS_STAGE1,
            out_classes=1,
            backbone_weights=Config.ENCODER_WEIGHTS,
        )

        # Stage 2: Takes original input + Stage 1 probability map (6 + 1 = 7 channels)
        self.stage2 = ResNetUNet(
            in_channels=Config.IN_CHANNELS_STAGE2,
            out_classes=1,
            backbone_weights=Config.ENCODER_WEIGHTS,
        )

    def forward(self, x):
        # --- Stage 1 ---
        logits1 = self.stage1(x)

        # Generate probability map for Stage 2 input
        # Detach to stop gradients flowing from Stage 2 back to Stage 1 through the input
        # Stage 1 is supervised by its own loss
        prob1 = torch.sigmoid(logits1).detach()

        # --- Stage 2 ---
        # Concatenate original input with Stage 1 prediction
        x2 = torch.cat([x, prob1], dim=1)
        logits2 = self.stage2(x2)

        return logits1, logits2
