import torch
import torch.nn as nn
from torchvision import models


class DepthInjector(nn.Module):
    """
    Non-linear MLP to project scalar depth to an embedding vector.
    Structure: Linear -> ReLU -> Linear.
    """

    def __init__(self, output_channels=32):
        super(DepthInjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(inplace=True), nn.Linear(32, output_channels)
        )

    def forward(self, z):
        # z shape: (B) or (B, 1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        z = z.float()
        return self.net(z)


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with internal width correction.
    Structure: 1x1 Conv -> 3x3 Deconv -> 1x1 Conv.
    Internal dimension is in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Width correction: internal dimension is in_channels // 4
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv: Reduce dimension
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv: Upsample
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
            # 1x1 Conv: Expand/Adjust dimension
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34 backbone with Wide-LinkNet decoder and Depth Concatenation.
    Designed for 128x128 input images (padded from 101x101).
    """

    def __init__(self, pretrained=True):
        super(ResNet34WideLinkNet, self).__init__()

        # Load Backbone
        resnet = models.resnet34(pretrained=pretrained)

        # Input Adaptation: Modify first conv to accept 1 channel
        # We sum the weights of the original RGB channels
        original_conv1 = resnet.conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
        )
        with torch.no_grad():
            self.conv1[0].weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Depth Injection
        self.depth_injector = DepthInjector(output_channels=32)

        # Decoder Blocks
        # Bottleneck: Encoder4 (512) + Depth (32) = 544
        self.decoder4 = DecoderBlock(544, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        # Final Head to restore resolution and channel count
        # Decoder1 output is 64x64. Need 128x128.
        self.final_head = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Logits
        )

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, 1, 128, 128)
            z: Depth tensor (B) or (B, 1)
        Returns:
            Logits (B, 1, 128, 128)
        """
        # --- Encoder ---
        # x: 128x128
        e0 = self.conv1(x)  # 64x64, 64ch
        e0_p = self.maxpool(e0)  # 32x32, 64ch
        e1 = self.encoder1(e0_p)  # 32x32, 64ch
        e2 = self.encoder2(e1)  # 16x16, 128ch
        e3 = self.encoder3(e2)  # 8x8, 256ch
        e4 = self.encoder4(e3)  # 4x4, 512ch

        # --- Depth Injection ---
        d_emb = self.depth_injector(z)  # (B, 32)
        # Expand depth embedding to spatial dimensions of bottleneck
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))  # (B, 32, 4, 4)

        # Concatenate
        center = torch.cat([e4, d_emb], dim=1)  # (B, 544, 4, 4)

        # --- Decoder with Additive Skips ---
        # LinkNet uses addition for skip connections

        d4 = self.decoder4(center)  # -> 8x8, 256ch
        d4 = d4 + e3

        d3 = self.decoder3(d4)  # -> 16x16, 128ch
        d3 = d3 + e2

        d2 = self.decoder2(d3)  # -> 32x32, 64ch
        d2 = d2 + e1

        d1 = self.decoder1(d2)  # -> 64x64, 64ch
        d1 = d1 + e0

        # --- Final Head ---
        out = self.final_head(d1)  # -> 128x128, 1ch

        return out
