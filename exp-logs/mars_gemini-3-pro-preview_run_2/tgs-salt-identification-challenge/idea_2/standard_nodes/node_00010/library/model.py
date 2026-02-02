import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block:
    1x1 Conv (Reduce) -> Transposed Conv (Upsample) -> 1x1 Conv (Expand)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels // 4,
                in_channels // 4,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthConditionedLinkNet(nn.Module):
    """
    LinkNet with ResNet34 Encoder and Lightweight Depth Injection.
    Cite Lesson 00008 (LinkNet Efficiency)
    Cite Lesson 00001 (Bottleneck Injection)
    Cite Lesson 00009 (Lightweight Concatenation)
    """

    def __init__(self, num_classes=1):
        super().__init__()
        # --- Encoder (ResNet34) ---
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Modify first layer to accept 1 channel
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.resnet.conv1.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))

        # Encoder Layers
        self.encoder1 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )  # 64, 64x64
        self.encoder2 = nn.Sequential(
            self.resnet.maxpool, self.resnet.layer1
        )  # 64, 32x32
        self.encoder3 = self.resnet.layer2  # 128, 16x16
        self.encoder4 = self.resnet.layer3  # 256, 8x8
        self.encoder5 = self.resnet.layer4  # 512, 4x4

        # --- Depth Injection ---
        # Cite Lesson 00009: Small embedding (32) concatenated directly
        self.depth_projector = nn.Sequential(nn.Linear(1, 32), nn.ReLU())

        # --- Decoder ---
        # Input to D5 is Bottleneck (512) + Depth (32) = 544
        self.decoder5 = DecoderBlock(512 + 32, 256)
        self.decoder4 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.decoder2 = DecoderBlock(64, 64)
        self.decoder1 = DecoderBlock(64, 32)  # Final upsample to 128x128

        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x, depth):
        # Encoder
        e1 = self.encoder1(x)  # 64x64, 64
        e2 = self.encoder2(e1)  # 32x32, 64
        e3 = self.encoder3(e2)  # 16x16, 128
        e4 = self.encoder4(e3)  # 8x8, 256
        e5 = self.encoder5(e4)  # 4x4, 512

        # Depth Injection
        d = self.depth_projector(depth).unsqueeze(-1).unsqueeze(-1)  # N, 32, 1, 1
        d = d.expand(-1, -1, e5.size(2), e5.size(3))  # N, 32, 4, 4
        bottleneck = torch.cat([e5, d], dim=1)  # 544 channels

        # Decoder (Additive Skip Connections)
        d5 = self.decoder5(bottleneck)
        d5 = d5 + e4  # 256

        d4 = self.decoder4(d5)
        d4 = d4 + e3  # 128

        d3 = self.decoder3(d4)
        d3 = d3 + e2  # 64

        d2 = self.decoder2(d3)
        d2 = d2 + e1  # 64

        d1 = self.decoder1(d2)  # 32 (128x128)

        return self.final(d1)
