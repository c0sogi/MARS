import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ContrailModel(nn.Module):
    """
    U-Net model with ConvNeXt-Small backbone for Contrail Identification.

    Re-implemented using `timm` for the encoder and a custom decoder to avoid
    dependency on `segmentation_models_pytorch`.
    """

    def __init__(self):
        super(ContrailModel, self).__init__()

        # Encoder: ConvNeXt-Small via timm
        # features_only=True returns feature maps from intermediate stages
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts from encoder
        # Typically [96, 192, 384, 768] for convnext_small
        enc_channels = self.encoder.feature_info.channels()

        # Decoder
        # We assume 4 stages from encoder with strides 4, 8, 16, 32

        # Stage 1: Stride 32 -> 16
        self.dec1 = DecoderBlock(enc_channels[3], enc_channels[2], 256)

        # Stage 2: Stride 16 -> 8
        self.dec2 = DecoderBlock(256, enc_channels[1], 128)

        # Stage 3: Stride 8 -> 4
        self.dec3 = DecoderBlock(128, enc_channels[0], 64)

        # Stage 4: Stride 4 -> 2 (No skip connection available from encoder usually)
        self.dec4 = DecoderBlock(64, 0, 32)

        # Stage 5: Stride 2 -> 1
        self.dec5 = DecoderBlock(32, 0, 16)

        # Final segmentation head
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 6, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch, 1, Height, Width).
        """
        features = self.encoder(x)
        # features: [f0(s4), f1(s8), f2(s16), f3(s32)]

        x = self.dec1(features[3], features[2])
        x = self.dec2(x, features[1])
        x = self.dec3(x, features[0])
        x = self.dec4(x)
        x = self.dec5(x)

        return self.head(x)
