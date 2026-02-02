import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A standard Convolutional Block consisting of two 3x3 convolutions,
    each followed by Batch Normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    """
    A lightweight 2D U-Net architecture.

    Args:
        in_channels (int): Number of input channels (default: 3 for Ash composite).
        out_channels (int): Number of output channels (default: 1 for binary mask).
        base_filters (int): Number of filters in the first encoder layer.
                            Reduced to 32 to keep the model lightweight (~7.7M params).
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super(SimpleUNet, self).__init__()

        # ----------------------------------------------------------------------
        # Encoder
        # ----------------------------------------------------------------------
        # Level 1
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 2
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 3
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 4
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----------------------------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------------------------
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # ----------------------------------------------------------------------
        # Decoder
        # ----------------------------------------------------------------------
        # Level 4
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels: Bottleneck (16*F) + Enc4 Skip (8*F)
        self.dec4 = ConvBlock(base_filters * 16 + base_filters * 8, base_filters * 8)

        # Level 3
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels: Dec4 (8*F) + Enc3 Skip (4*F)
        self.dec3 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)

        # Level 2
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels: Dec3 (4*F) + Enc2 Skip (2*F)
        self.dec2 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)

        # Level 1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels: Dec2 (2*F) + Enc1 Skip (1*F)
        self.dec1 = ConvBlock(base_filters * 2 + base_filters, base_filters)

        # ----------------------------------------------------------------------
        # Output
        # ----------------------------------------------------------------------
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with Skip Connections
        d4 = self.up4(b)
        # Concatenate along channel dimension (dim 1)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output Head
        out = self.final_conv(d1)
        return torch.sigmoid(out)
