import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Unet(nn.Module):
    def __init__(self, encoder_name, in_channels, classes):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, features_only=True, in_chans=in_channels
        )

        # Get channel counts from encoder features
        enc_channels = self.encoder.feature_info.channels()

        # Decoder construction (Simple U-Net style)
        # Using last 5 features: f1(s2), f2(s4), f3(s8), f4(s16), f5(s32)
        self.up1 = DecoderBlock(enc_channels[-1], enc_channels[-2], enc_channels[-2])
        self.up2 = DecoderBlock(enc_channels[-2], enc_channels[-3], enc_channels[-3])
        self.up3 = DecoderBlock(enc_channels[-3], enc_channels[-4], enc_channels[-4])
        self.up4 = DecoderBlock(enc_channels[-4], enc_channels[-5], enc_channels[-5])

        self.final_conv = nn.Conv2d(enc_channels[-5], classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)

        x = self.up1(features[-1], features[-2])
        x = self.up2(x, features[-3])
        x = self.up3(x, features[-4])
        x = self.up4(x, features[-5])

        # Final upsampling (stride 2 -> 1)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return self.final_conv(x)


def build_model(
    encoder_name=Config.ENCODER,
    encoder_weights=Config.ENCODER_WEIGHTS,
    in_channels=Config.IN_CHANNELS,
    classes=Config.CLASSES,
    activation=Config.ACTIVATION,
):
    """
    Constructs a U-Net architecture using timm encoder.
    Replaces segmentation_models_pytorch dependency.
    """
    # Ensure encoder name is timm-compatible
    encoder_name = encoder_name.replace("-", "_")
    return Unet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
