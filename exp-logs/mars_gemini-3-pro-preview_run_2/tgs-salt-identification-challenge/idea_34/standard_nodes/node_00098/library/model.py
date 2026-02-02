import torch
import torch.nn as nn
from torchvision import models


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Structure: 1x1 Conv (reduce) -> 3x3 Deconv (upsample) -> 1x1 Conv (expand).
    Internal width is in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        mid_channels = in_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34Encoder(nn.Module):
    """
    ResNet34 Encoder modified for 1-channel input.
    Returns features from 5 levels:
    e1: (64, 64, 64) - after relu
    e2: (64, 32, 32) - layer1
    e3: (128, 16, 16) - layer2
    e4: (256, 8, 8) - layer3
    e5: (512, 4, 4) - layer4
    (Assuming 128x128 input)
    """

    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        # Use modern weights parameter
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first convolution for 1-channel input
        # Sum weights across channel dimension to preserve intensity information
        original_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            backbone.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        # Input: (B, 1, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        e1 = self.relu(x)

        x = self.maxpool(e1)
        e2 = self.layer1(x)
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)

        return e1, e2, e3, e4, e5


class SaltModel(nn.Module):
    """
    ResNet34-WideLinkNet with Explicit Depth Injection.
    Projects scalar depth to feature maps and concatenates with bottleneck.
    """

    def __init__(self):
        super(SaltModel, self).__init__()
        self.encoder = ResNet34Encoder()

        # Depth Injection Module
        self.depth_channels = 64
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.depth_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder Blocks
        # Block 4: Input e5 (512) + depth (64) = 576. Output 256 (matches e4).
        self.dec4 = DecoderBlock(512 + self.depth_channels, 256)
        # Block 3: Input 256. Output 128 (matches e3).
        self.dec3 = DecoderBlock(256, 128)
        # Block 2: Input 128. Output 64 (matches e2).
        self.dec2 = DecoderBlock(128, 64)
        # Block 1: Input 64. Output 64 (matches e1).
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsampling to restore original resolution (e.g. 64x64 -> 128x128)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, depth):
        # Encoder
        e1, e2, e3, e4, e5 = self.encoder(x)

        # Depth Injection
        # depth: (B, 1) -> (B, 64) -> (B, 64, 1, 1) -> (B, 64, H_e5, W_e5)
        d = self.depth_mlp(depth)
        d = d.view(-1, self.depth_channels, 1, 1)
        d = d.expand(-1, -1, e5.size(2), e5.size(3))

        e5_aug = torch.cat([e5, d], dim=1)

        # Decoder with Additive Skip Connections
        d4 = self.dec4(e5_aug)
        d4 = d4 + e4

        d3 = self.dec3(d4)
        d3 = d3 + e3

        d2 = self.dec2(d3)
        d2 = d2 + e2

        d1 = self.dec1(d2)
        d1 = d1 + e1

        # Final Prediction
        out = self.final_conv(d1)
        return out
