import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with additive skip connections support.
    Structure: 1x1 Conv -> TransposeConv (Upsample) -> 1x1 Conv.
    Internal width is set to in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # "Wide" strategy: internal dimension based on input
        internal_channels = in_channels // 4

        # Ensure internal channels is at least a small number to avoid bottlenecks
        if internal_channels < 16:
            internal_channels = 16

        self.block = nn.Sequential(
            # 1x1 Conv to reduce dimensions
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # Transposed Conv for upsampling
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
            # 1x1 Conv to expand dimensions
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    FP32-Stabilized Multi-Task Specialist Model.
    Backbone: ResNet34 (1-channel input).
    Neck: Depth Injection (Concat).
    Decoder: Wide-LinkNet with Additive Skips.
    Head: Segmentation + Auxiliary Depth Regression.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ResNet34
        # Load pretrained weights
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Input Adapter: Convert first layer to 1-channel
        # Sum weights across RGB channels: (64, 3, 7, 7) -> (64, 1, 7, 7)
        original_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.encoder_conv1 = new_conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        self.encoder_layer1 = resnet.layer1  # 64
        self.encoder_layer2 = resnet.layer2  # 128
        self.encoder_layer3 = resnet.layer3  # 256
        self.encoder_layer4 = resnet.layer4  # 512

        # 2. Depth Injector
        # Projects scalar z -> 32 channels
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(1, 32),  # Wait, input to this linear is 16
        )
        # Correcting the definition above
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 32)
        )

        # 3. Auxiliary Head
        # Predicts depth from bottleneck features
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # 4. Decoder
        # Bottleneck input: 512 (encoder) + 32 (depth) = 544

        # Block 4: 544 -> 256 (Matches layer3)
        self.decoder4 = DecoderBlock(544, 256)

        # Block 3: 256 -> 128 (Matches layer2)
        self.decoder3 = DecoderBlock(256, 128)

        # Block 2: 128 -> 64 (Matches layer1)
        self.decoder2 = DecoderBlock(128, 64)

        # Block 1: 64 -> 64 (Matches conv1/bn1/relu output)
        self.decoder1 = DecoderBlock(64, 64)

        # Final Upsample: 64 -> 32 -> 1
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x, z):
        """
        Args:
            x: Images (B, 1, H, W)
            z: Depths (B, 1)
        """
        # --- Encoder ---
        # Stem
        x0 = self.encoder_conv1(x)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0)  # Shape: (B, 64, H/2, W/2)

        x_pool = self.encoder_maxpool(x0)  # Shape: (B, 64, H/4, W/4)

        # Blocks
        e1 = self.encoder_layer1(x_pool)  # (B, 64, H/4, W/4)
        e2 = self.encoder_layer2(e1)  # (B, 128, H/8, W/8)
        e3 = self.encoder_layer3(e2)  # (B, 256, H/16, W/16)
        e4 = self.encoder_layer4(e3)  # (B, 512, H/32, W/32)

        # --- Auxiliary Task ---
        aux_pred = self.aux_head(e4)

        # --- Depth Injection ---
        # Process depth
        z_emb = self.depth_mlp(z)  # (B, 32)
        # Expand spatially to match e4
        z_emb = z_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        z_emb = z_emb.expand(-1, -1, e4.size(2), e4.size(3))  # (B, 32, H/32, W/32)

        # Concatenate
        bottleneck = torch.cat([e4, z_emb], dim=1)  # (B, 544, H/32, W/32)

        # --- Decoder (LinkNet Style: Additive Skips) ---

        # Block 4
        d4 = self.decoder4(bottleneck)  # (B, 256, H/16, W/16)
        d4 = d4 + e3  # Additive Skip

        # Block 3
        d3 = self.decoder3(d4)  # (B, 128, H/8, W/8)
        d3 = d3 + e2  # Additive Skip

        # Block 2
        d2 = self.decoder2(d3)  # (B, 64, H/4, W/4)
        d2 = d2 + e1  # Additive Skip

        # Block 1
        d1 = self.decoder1(d2)  # (B, 64, H/2, W/2)
        d1 = d1 + x0  # Additive Skip (Stem features)

        # Final Output
        logits = self.final_upsample(d1)  # (B, 1, H, W)

        return logits, aux_pred
