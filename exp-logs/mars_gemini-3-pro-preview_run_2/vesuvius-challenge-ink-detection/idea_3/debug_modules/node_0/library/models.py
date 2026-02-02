import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import ENCODER_NAME, ENCODER_WEIGHTS, IN_CHANNELS, CLASSES


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Calculate input channels for the convolution (input + skip)
        conv_in_channels = in_channels + skip_channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Handle potential shape mismatch due to rounding in encoder pooling
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class ResNet34UNet(nn.Module):
    """
    U-Net with ResNet34 Encoder.
    """

    def __init__(self, in_channels=3, classes=1, pretrained=True):
        super(ResNet34UNet, self).__init__()

        # --- Encoder (ResNet34) ---
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Adapt first layer if input channels differ from 3 (RGB)
        if in_channels != 3:
            self.encoder_conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Copy weights from original if possible (averaging or slicing),
            # but here we just initialize randomly if channels don't match.
            # For this task, IN_CHANNELS is 3, so this block usually skipped.
        else:
            self.encoder_conv1 = resnet.conv1

        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        self.encoder_layer1 = resnet.layer1  # 64 channels, H/4
        self.encoder_layer2 = resnet.layer2  # 128 channels, H/8
        self.encoder_layer3 = resnet.layer3  # 256 channels, H/16
        self.encoder_layer4 = resnet.layer4  # 512 channels, H/32

        # --- Decoder ---
        # Layer 4 (512) -> Upsample + Layer 3 (256) -> 256
        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )

        # Layer 3 (256) -> Upsample + Layer 2 (128) -> 128
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Layer 2 (128) -> Upsample + Layer 1 (64) -> 64
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Layer 1 (64) -> Upsample + Layer 0 (64) -> 64
        # Layer 0 is the output of Conv1+BN+ReLU (before maxpool), which has 64 channels and size H/2
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Final Upsample: H/2 -> H. No skip connection from raw input usually in this design.
        self.decoder0 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        # --- Head ---
        self.segmentation_head = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Stem
        x0 = self.encoder_conv1(x)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)  # H/2, 64 ch

        x1 = self.encoder_maxpool(x0)  # H/4, 64 ch
        x1 = self.encoder_layer1(x1)  # H/4, 64 ch

        x2 = self.encoder_layer2(x1)  # H/8, 128 ch
        x3 = self.encoder_layer3(x2)  # H/16, 256 ch
        x4 = self.encoder_layer4(x3)  # H/32, 512 ch

        # --- Decoder ---
        d4 = self.decoder4(x4, x3)  # -> H/16
        d3 = self.decoder3(d4, x2)  # -> H/8
        d2 = self.decoder2(d3, x1)  # -> H/4
        d1 = self.decoder1(d2, x0)  # -> H/2
        d0 = self.decoder0(d1)  # -> H

        # --- Head ---
        out = self.segmentation_head(d0)

        return out


def build_model():
    """
    Factory function to create the model based on config.
    """
    pretrained = ENCODER_WEIGHTS == "imagenet"
    model = ResNet34UNet(
        in_channels=IN_CHANNELS, classes=CLASSES, pretrained=pretrained
    )
    return model
