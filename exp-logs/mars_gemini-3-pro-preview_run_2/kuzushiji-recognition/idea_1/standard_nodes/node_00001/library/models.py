import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config, seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, we use normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Features from the previous decoder layer (to be upsampled)
            x2: Features from the encoder skip connection (to be concatenated)
        """
        x1 = self.up(x1)

        # Input is CHW. Handle padding in case of odd dimensions/mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegmentationUNet(nn.Module):
    """
    U-Net with ResNet18 Encoder for Binary Segmentation.
    """

    def __init__(self, n_classes=1, bilinear=True):
        super(SegmentationUNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load ResNet18 backbone
        # Attempt to use modern weights API, fallback to pretrained=True, then random init
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
        except (AttributeError, ImportError):
            try:
                self.resnet = models.resnet18(pretrained=True)
            except:
                self.resnet = models.resnet18(pretrained=False)

        # Encoder Layers (ResNet18)
        # Input layer
        self.inc = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )  # Output: 64 channels, Stride 2

        self.pool = self.resnet.maxpool
        self.layer1 = (
            self.resnet.layer1
        )  # Output: 64 channels, Stride 4 (relative to input)
        self.layer2 = self.resnet.layer2  # Output: 128 channels, Stride 8
        self.layer3 = self.resnet.layer3  # Output: 256 channels, Stride 16
        self.layer4 = self.resnet.layer4  # Output: 512 channels, Stride 32

        # Decoder Layers
        # Up(in_channels, out_channels) where in_channels = skip_channels + upsampled_channels
        # x5 (512) + x4 (256) -> 256
        self.up1 = Up(512 + 256, 256, bilinear)
        # d1 (256) + x3 (128) -> 128
        self.up2 = Up(256 + 128, 128, bilinear)
        # d2 (128) + x2 (64) -> 64
        self.up3 = Up(128 + 64, 64, bilinear)
        # d3 (64) + x1 (64) -> 64
        self.up4 = Up(64 + 64, 64, bilinear)

        # Final output layer
        # Upsample from Stride 2 to Stride 1 (Original Resolution)
        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # Stride 2
        x2 = self.pool(x1)  # Stride 4
        x2 = self.layer1(x2)  # Stride 4 (ResNet layer1 doesn't downsample)
        x3 = self.layer2(x2)  # Stride 8
        x4 = self.layer3(x3)  # Stride 16
        x5 = self.layer4(x4)  # Stride 32

        # Decoder
        d1 = self.up1(x5, x4)  # Stride 16
        d2 = self.up2(d1, x3)  # Stride 8
        d3 = self.up3(d2, x2)  # Stride 4
        d4 = self.up4(d3, x1)  # Stride 2

        # Final Upsample and Prediction
        out = self.up_final(d4)  # Stride 1
        logits = self.outc(out)

        return logits


class CharacterClassifier(nn.Module):
    """
    ResNet18 based Classifier for Kuzushiji Characters.
    """

    def __init__(self, num_classes):
        super(CharacterClassifier, self).__init__()

        # Load ResNet18 backbone
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
        except (AttributeError, ImportError):
            try:
                self.resnet = models.resnet18(pretrained=True)
            except:
                self.resnet = models.resnet18(pretrained=False)

        # Replace the final Fully Connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
