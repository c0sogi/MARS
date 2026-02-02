import torch
import torch.nn as nn
from torchvision import models


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block.
    Performs efficient upsampling using a bottleneck approach:
    1. 1x1 Conv (reduce channels)
    2. 3x3 Transposed Conv (upsample)
    3. 1x1 Conv (expand channels)
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # LinkNet design typically reduces channels by 4 internally for efficiency
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv to reduce dimensions
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv to upsample
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv to expand dimensions to match the skip connection
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LinkNetResNet18(nn.Module):
    """
    LinkNet architecture with a ResNet-34 encoder.
    Cite solution_lesson_node_00006: Scaling Model Capacity
    """

    def __init__(self, in_channels=3, classes=1):
        super(LinkNetResNet18, self).__init__()

        # --- Encoder ---
        # Load pre-trained ResNet34
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            self.encoder = models.resnet34(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.encoder = models.resnet34(pretrained=True)

        # Handle non-standard input channels (e.g., if input is not RGB)
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove unused layers to save memory
        del self.encoder.avgpool
        del self.encoder.fc

        # --- Decoder ---
        # ResNet18 Feature Channels:
        # Layer 4: 512 (Stride 32)
        # Layer 3: 256 (Stride 16)
        # Layer 2: 128 (Stride 8)
        # Layer 1: 64  (Stride 4)
        # Stem:    64  (Stride 2)

        # Decoder 4: Input 512 -> Output 256 (Adds with Layer 3)
        self.decoder4 = DecoderBlock(512, 256)

        # Decoder 3: Input 256 -> Output 128 (Adds with Layer 2)
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: Input 128 -> Output 64 (Adds with Layer 1)
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: Input 64 -> Output 64 (Adds with Stem)
        self.decoder1 = DecoderBlock(64, 64)

        # --- Final Head ---
        # Upsample from Stride 2 (Stem) to Stride 1 (Original Resolution)
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, classes, kernel_size=3, padding=1)

    def forward(self, x):
        # --- Encoder Forward Pass ---
        # Stem
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        e1 = self.encoder.relu(x)  # (B, 64, H/2, W/2)

        x = self.encoder.maxpool(e1)  # (B, 64, H/4, W/4)

        # ResNet Layers
        e2 = self.encoder.layer1(x)  # (B, 64, H/4, W/4)
        e3 = self.encoder.layer2(e2)  # (B, 128, H/8, W/8)
        e4 = self.encoder.layer3(e3)  # (B, 256, H/16, W/16)
        e5 = self.encoder.layer4(e4)  # (B, 512, H/32, W/32)

        # --- Decoder Forward Pass ---
        # LinkNet adds the encoder feature map to the decoder output

        # Block 4
        d4 = self.decoder4(e5)  # Upsample to H/16
        d4 = d4 + e4  # Add skip connection

        # Block 3
        d3 = self.decoder3(d4)  # Upsample to H/8
        d3 = d3 + e3  # Add skip connection

        # Block 2
        d2 = self.decoder2(d3)  # Upsample to H/4
        d2 = d2 + e2  # Add skip connection

        # Block 1
        d1 = self.decoder1(d2)  # Upsample to H/2
        d1 = d1 + e1  # Add skip connection

        # --- Final Prediction ---
        out = self.final_deconv(d1)  # Upsample to H
        out = self.final_conv(out)  # (B, classes, H, W)

        return out
