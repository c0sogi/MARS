import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionBlock(nn.Module):
    """
    Attention Gate to filter features from the skip connection.
    Uses the gating signal from the decoder to suppress irrelevant regions
    in the skip connection features from the encoder.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from decoder, coarser resolution)
        # x: skip connection (from encoder, finer resolution)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Resize g1 to match x1 spatial dimensions if necessary
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder block enhanced with Attention Gates.
    Performs: Upsample -> Attention Gate -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Attention Block
        # F_g = in_channels (upsampled features)
        # F_l = skip_channels (encoder features)
        # F_int = intermediate channels (usually half of input)
        self.attn = AttentionBlock(
            F_g=in_channels, F_l=skip_channels, F_int=in_channels // 2
        )

        # Double Conv Block after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attn(g=x, x=skip)

        # Ensure shapes match exactly before concatenation (handling odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionUNetResNet34(nn.Module):
    """
    Anatomy-Aware Attention U-Net with ResNet-34 Backbone.
    Modified to accept 4-channel input (RGB + Anatomical Mask).
    """

    def __init__(self, in_channels=4, num_classes=1, pretrained=True):
        super(AttentionUNetResNet34, self).__init__()

        # 1. Encoder (ResNet34)
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Modify first conv layer to accept 'in_channels' (e.g., 4)
        original_conv1 = resnet.conv1
        self.encoder_conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize weights for the new first layer
        with torch.no_grad():
            if in_channels >= 3:
                # Copy RGB weights
                self.encoder_conv1.weight[:, :3] = original_conv1.weight
            if in_channels > 3:
                # Initialize additional channels (e.g., anatomical mask) with mean of RGB weights
                # This provides a reasonable initialization compared to random noise
                self.encoder_conv1.weight[:, 3:] = torch.mean(
                    original_conv1.weight, dim=1, keepdim=True
                )

        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        self.encoder_layer1 = resnet.layer1  # 64 channels
        self.encoder_layer2 = resnet.layer2  # 128 channels
        self.encoder_layer3 = resnet.layer3  # 256 channels
        self.encoder_layer4 = resnet.layer4  # 512 channels

        # 2. Decoder
        # D4: Input from Layer4 (512), Skip from Layer3 (256)
        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )

        # D3: Input from D4 (256), Skip from Layer2 (128)
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # D2: Input from D3 (128), Skip from Layer1 (64)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # D1: Input from D2 (64), Skip from Conv1 output (64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # 3. Final Output
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # x: (B, 4, H, W)
        x0 = self.encoder_conv1(x)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)  # (B, 64, H/2, W/2)

        x_pool = self.encoder_maxpool(x0)  # (B, 64, H/4, W/4)

        x1 = self.encoder_layer1(x_pool)  # (B, 64, H/4, W/4)
        x2 = self.encoder_layer2(x1)  # (B, 128, H/8, W/8)
        x3 = self.encoder_layer3(x2)  # (B, 256, H/16, W/16)
        x4 = self.encoder_layer4(x3)  # (B, 512, H/32, W/32) -> Bottleneck

        # --- Decoder with Attention ---
        d4 = self.decoder4(x4, x3)  # -> (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, x2)  # -> (B, 128, H/8, W/8)
        d2 = self.decoder2(d3, x1)  # -> (B, 64, H/4, W/4)
        d1 = self.decoder1(d2, x0)  # -> (B, 64, H/2, W/2)

        # --- Final Classification ---
        out = self.final_upsample(d1)  # -> (B, 64, H, W)
        out = self.final_conv(out)  # -> (B, num_classes, H, W)

        return out
