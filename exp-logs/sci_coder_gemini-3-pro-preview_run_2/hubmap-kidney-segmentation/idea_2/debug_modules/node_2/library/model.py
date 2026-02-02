import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block: Upsample -> Concat with Skip -> ConvBlock
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle potential padding issues if shapes aren't perfectly divisible
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AnatomyAwareUNetPlusPlus(nn.Module):
    """
    Anatomy-Aware U-Net Model (Re-implemented with torchvision).

    This replaces the 'segmentation_models_pytorch' dependency with a custom
    U-Net implementation using a ResNet34 encoder from torchvision.

    It accepts 4-channel inputs:
    1-3. RGB Image Channels
    4.   Binary Anatomical Mask
    """

    def __init__(self):
        super(AnatomyAwareUNetPlusPlus, self).__init__()

        # Load Pre-trained Encoder
        weights = ResNet34_Weights.DEFAULT
        self.encoder = resnet34(weights=weights)

        # Modify the input layer to accept 4 channels
        self._adapt_input_layer()

        # Encoder Layers (ResNet34)
        # layer0: conv1, bn1, relu, maxpool
        self.enc0 = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu
        )
        self.pool = self.encoder.maxpool
        self.enc1 = self.encoder.layer1  # 64 channels
        self.enc2 = self.encoder.layer2  # 128 channels
        self.enc3 = self.encoder.layer3  # 256 channels
        self.enc4 = self.encoder.layer4  # 512 channels

        # Decoder Layers
        # Center: 512 channels
        # Dec4: 512 + 256 (skip) -> 256
        self.dec4 = DecoderBlock(512, 256, 256)
        # Dec3: 256 + 128 (skip) -> 128
        self.dec3 = DecoderBlock(256, 128, 128)
        # Dec2: 128 + 64 (skip) -> 64
        self.dec2 = DecoderBlock(128, 64, 64)
        # Dec1: 64 + 64 (skip from enc0) -> 32
        self.dec1 = DecoderBlock(64, 64, 32)

        # Final Convolution
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, Config.CLASSES, 1),
        )

    def _adapt_input_layer(self):
        """
        Replaces the first convolutional layer of the encoder to accept
        Config.IN_CHANNELS (4) instead of the default 3.
        """
        old_conv = self.encoder.conv1
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights: Copy RGB, average for Anatomy
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = torch.mean(
                old_conv.weight, dim=1, keepdim=True
            )
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        self.encoder.conv1 = new_conv

    def forward(self, x):
        """
        Forward pass.
        Returns logits (no sigmoid).
        """
        # Encoder
        x0 = self.enc0(x)  # (B, 64, H/2, W/2)
        x_pool = self.pool(x0)  # (B, 64, H/4, W/4)

        x1 = self.enc1(x_pool)  # (B, 64, H/4, W/4)
        x2 = self.enc2(x1)  # (B, 128, H/8, W/8)
        x3 = self.enc3(x2)  # (B, 256, H/16, W/16)
        x4 = self.enc4(x3)  # (B, 512, H/32, W/32)

        # Decoder
        d4 = self.dec4(x4, x3)  # -> (B, 256, H/16, W/16)
        d3 = self.dec3(d4, x2)  # -> (B, 128, H/8, W/8)
        d2 = self.dec2(d3, x1)  # -> (B, 64, H/4, W/4)
        d1 = self.dec1(d2, x0)  # -> (B, 32, H/2, W/2)

        # Final Output
        out = self.final_conv(d1)  # -> (B, 1, H, W)

        return out
