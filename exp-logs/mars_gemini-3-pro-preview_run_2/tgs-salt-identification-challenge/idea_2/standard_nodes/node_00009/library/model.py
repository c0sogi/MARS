import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block:
    Conv1x1 (reduce) -> TransposedConv (upsample) -> Conv1x1 (expand)
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
    LinkNet with ResNet34 Encoder and Depth Injection.
    Uses additive skip connections for efficiency.
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

        # --- Depth Injection ---
        self.depth_projector = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid(),
        )
        # Fusion after concatenation
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        # ResNet34 channels:
        # layer4 (x4): 512
        # layer3 (x3): 256
        # layer2 (x2): 128
        # layer1 (x1): 64
        # conv1 (x0): 64

        self.dec4 = DecoderBlock(512, 256)  # Out: 256 (matches x3)
        self.dec3 = DecoderBlock(256, 128)  # Out: 128 (matches x2)
        self.dec2 = DecoderBlock(128, 64)  # Out: 64 (matches x1)
        self.dec1 = DecoderBlock(64, 64)  # Out: 64 (matches x0)

        # Final upsampling block (64 -> 32 -> num_classes)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, depth):
        # --- Encoder ---
        # x: (N, 1, 128, 128)
        x0 = self.resnet.conv1(x)  # (N, 64, 64, 64)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)

        x1 = self.resnet.maxpool(x0)  # (N, 64, 32, 32)
        x1 = self.resnet.layer1(x1)  # (N, 64, 32, 32)

        x2 = self.resnet.layer2(x1)  # (N, 128, 16, 16)
        x3 = self.resnet.layer3(x2)  # (N, 256, 8, 8)
        x4 = self.resnet.layer4(x3)  # (N, 512, 4, 4)

        # --- Depth Injection ---
        d = self.depth_projector(depth).unsqueeze(-1).unsqueeze(-1)  # (N, 512, 1, 1)
        d = d.expand_as(x4)
        x4 = self.bottleneck_fusion(torch.cat([x4, d], dim=1))

        # --- Decoder (Additive Skips) ---
        d4 = self.dec4(x4) + x3
        d3 = self.dec3(d4) + x2
        d2 = self.dec2(d3) + x1
        d1 = self.dec1(d2) + x0

        d0 = self.dec0(d1)
        return self.final(d0)
