import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DepthInjector(nn.Module):
    """
    Projects a scalar depth value into a feature map and concatenates it.
    """

    def __init__(self, output_channels=64):
        super().__init__()
        self.output_channels = output_channels
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_channels),
        )

    def forward(self, z, feature_map):
        """
        Args:
            z: (B, 1) scalar depth tensor.
            feature_map: (B, C, H, W) feature map.
        Returns:
            (B, C + output_channels, H, W) tensor.
        """
        # Project z
        z_emb = self.mlp(z)  # (B, output_channels)

        # Reshape and expand to match feature map spatial dimensions
        z_emb = z_emb.view(z_emb.size(0), self.output_channels, 1, 1)
        z_emb = z_emb.expand(-1, -1, feature_map.size(2), feature_map.size(3))

        # Concatenate along channel dimension
        return torch.cat([feature_map, z_emb], dim=1)


class AuxiliaryHead(nn.Module):
    """
    Auxiliary head to predict depth from texture features.
    """

    def __init__(self, in_channels=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.mlp(x)


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Transposed Convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Wide-LinkNet: internal width = in_channels // 4
        # Ensure internal width is at least 16 to avoid collapse on small inputs
        internal_channels = max(in_channels // 4, 16)

        # 1x1 Conv to reduce dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
        )

        # 3x3 Transpose Conv to upsample
        self.deconv2 = nn.Sequential(
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
        )

        # 1x1 Conv to expand dimensions (or match target)
        self.conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        return x


class ResNet34WideLinkNetMTL(nn.Module):
    """
    ResNet34 Encoder + Wide-LinkNet Decoder with Multi-Task Learning (Aux Depth)
    and Conditional Depth Injection.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Encoder (ResNet34)
        # =====================================================================
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first conv layer to accept 1 channel (sum weights)
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        # Extract layers
        self.encoder_stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder_pool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1  # 64 channels
        self.encoder_layer2 = resnet.layer2  # 128 channels
        self.encoder_layer3 = resnet.layer3  # 256 channels
        self.encoder_layer4 = resnet.layer4  # 512 channels

        # =====================================================================
        # 2. Auxiliary Head (Depth Prediction)
        # =====================================================================
        self.aux_head = AuxiliaryHead(in_channels=512)

        # =====================================================================
        # 3. Depth Injector
        # =====================================================================
        self.inject_depth = Config.INJECT_DEPTH
        self.injector_channels = 64
        self.injector = DepthInjector(output_channels=self.injector_channels)

        # =====================================================================
        # 4. Decoder (Wide-LinkNet)
        # =====================================================================
        # Decoder Channels from Config: [256, 128, 64, 32, 16]

        # Block 1: Bottleneck -> Layer 3
        # Input: 512 + 64 (if injected) -> Output: 256
        in_ch_1 = 512 + (self.injector_channels if self.inject_depth else 0)
        self.dec1 = DecoderBlock(in_ch_1, 256)

        # Block 2: Layer 3 -> Layer 2
        # Input: 256 -> Output: 128
        self.dec2 = DecoderBlock(256, 128)

        # Block 3: Layer 2 -> Layer 1
        # Input: 128 -> Output: 64
        self.dec3 = DecoderBlock(128, 64)

        # Block 4: Layer 1 -> Stem
        # Input: 64 -> Output: 32
        self.dec4 = DecoderBlock(64, 32)

        # Projection for Stem Skip Connection (64 -> 32) to allow additive skip
        self.skip_proj4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32)
        )

        # Block 5: Stem -> Final Resolution
        # Input: 32 -> Output: 16
        self.dec5 = DecoderBlock(32, 16)

        # Final Convolution
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, depth=None):
        """
        Args:
            x: (B, 1, H, W) Input image.
            depth: (B, 1) Normalized depth value (optional).
        Returns:
            dict: {'mask': logits, 'depth': predicted_depth}
        """
        # --- Encoder Forward ---
        e0 = self.encoder_stem(x)  # (B, 64, H/2, W/2)
        e0_pool = self.encoder_pool(e0)  # (B, 64, H/4, W/4)
        e1 = self.encoder_layer1(e0_pool)  # (B, 64, H/4, W/4)
        e2 = self.encoder_layer2(e1)  # (B, 128, H/8, W/8)
        e3 = self.encoder_layer3(e2)  # (B, 256, H/16, W/16)
        e4 = self.encoder_layer4(e3)  # (B, 512, H/32, W/32)

        # --- Auxiliary Task ---
        pred_depth = self.aux_head(e4)

        # --- Conditional Injection ---
        center = e4
        if self.inject_depth and depth is not None:
            center = self.injector(depth, e4)

        # --- Decoder Forward (Additive Skips) ---

        # Block 1: Upsample Center, Add e3
        d1 = self.dec1(center)
        d1 = d1 + e3

        # Block 2: Upsample d1, Add e2
        d2 = self.dec2(d1)
        d2 = d2 + e2

        # Block 3: Upsample d2, Add e1
        d3 = self.dec3(d2)
        d3 = d3 + e1

        # Block 4: Upsample d3, Add Projected e0
        d4 = self.dec4(d3)
        s4 = self.skip_proj4(e0)
        d4 = d4 + s4

        # Block 5: Upsample d4 (No skip from image)
        d5 = self.dec5(d4)

        # Final Logits
        mask = self.final_conv(d5)

        return {"mask": mask, "depth": pred_depth}
