import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class DepthInjector(nn.Module):
    """
    Projects scalar depth z into a spatial feature map to be concatenated with encoder features.
    Architecture: Linear -> ReLU -> Linear -> Reshape -> Broadcast
    """

    def __init__(self, output_channels=32):
        super(DepthInjector, self).__init__()
        # Non-Linear MLP to project scalar depth to embedding
        self.mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, output_channels)
        )

    def forward(self, z, feature_map_shape):
        """
        Args:
            z: (B,) or (B, 1) scalar depth values.
            feature_map_shape: Tuple/Size of the target feature map (B, C, H, W).
        Returns:
            Tensor of shape (B, output_channels, H, W).
        """
        if z.dim() == 1:
            z = z.view(-1, 1)

        # Project to embedding: (B, 1) -> (B, output_channels)
        embedding = self.mlp(z)

        # Reshape to (B, C, 1, 1)
        embedding = embedding.unsqueeze(2).unsqueeze(3)

        # Expand to match spatial dimensions of the feature map
        h, w = feature_map_shape[2], feature_map_shape[3]
        return embedding.expand(-1, -1, h, w)


class DecoderBlock(nn.Module):
    """
    LinkNet-style decoder block with internal width correction.
    Structure: 1x1 Conv -> 3x3 Transpose Conv -> 1x1 Conv
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Width Correction: Calculate internal dimension as in_channels // 4
        # This preserves more semantic information compared to standard out_channels // 4
        mid_channels = max(in_channels // 4, 16)

        self.block = nn.Sequential(
            # 1x1 Conv to reduce/adjust channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transpose Conv for upsampling (Stride 2)
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
            # 1x1 Conv to expand to output channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    Soft-Distilled Robust Wide-LinkNet Architecture.
    Backbone: ResNet34 (Modified for 1-channel input).
    Bottleneck: Depth Injection + Concatenation.
    Decoder: Wide-LinkNet blocks with Additive Skip Connections.
    """

    def __init__(self, pretrained=True):
        super(ResNet34WideLinkNet, self).__init__()

        # 1. Backbone: ResNet34
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.resnet = resnet34(weights=weights)

        # 2. Input Adaptation: Modify first layer for 1-channel input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Sum the weights across the channel dimension to preserve pretrained feature detectors
            with torch.no_grad():
                self.resnet.conv1.weight.copy_(
                    original_conv1.weight.sum(dim=1, keepdim=True)
                )

        # Extract Encoder Layers
        # Encoder 0: (B, 64, H/2, W/2) - After Conv1, BN, ReLU
        self.encoder0 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )
        # Encoder 1: (B, 64, H/4, W/4) - After MaxPool + Layer1
        self.encoder1 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)
        # Encoder 2: (B, 128, H/8, W/8) - Layer2
        self.encoder2 = self.resnet.layer2
        # Encoder 3: (B, 256, H/16, W/16) - Layer3
        self.encoder3 = self.resnet.layer3
        # Encoder 4: (B, 512, H/32, W/32) - Layer4
        self.encoder4 = self.resnet.layer4

        # 3. Depth Injection
        self.depth_injector = DepthInjector(output_channels=32)

        # 4. Decoder
        # Bottleneck Input: Enc4 (512) + Depth (32) = 544 channels
        # Decoder 4: 544 -> 256 (Matches Enc3 channels for addition)
        self.decoder4 = DecoderBlock(544, 256)

        # Decoder 3: 256 -> 128 (Matches Enc2 channels)
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: 128 -> 64 (Matches Enc1 channels)
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: 64 -> 64 (Matches Enc0 channels)
        self.decoder1 = DecoderBlock(64, 64)

        # Decoder 0: 64 -> 32 (Final Upsample to original resolution)
        self.decoder0 = DecoderBlock(64, 32)

        # Final Classification Head
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, 1, H, W)
            z: Depth tensor (B,) or (B, 1)
        Returns:
            Logits tensor (B, 1, H, W)
        """
        # --- Encoder ---
        e0 = self.encoder0(x)  # 64, H/2
        e1 = self.encoder1(e0)  # 64, H/4
        e2 = self.encoder2(e1)  # 128, H/8
        e3 = self.encoder3(e2)  # 256, H/16
        e4 = self.encoder4(e3)  # 512, H/32

        # --- Bottleneck & Depth Injection ---
        # Project depth to match e4 spatial dim
        d = self.depth_injector(z, e4.shape)  # 32, H/32

        # Concatenate features and depth embedding
        bottleneck = torch.cat([e4, d], dim=1)  # 544, H/32

        # --- Decoder with Additive Skip Connections ---
        # Block 4
        d4 = self.decoder4(bottleneck)  # -> 256, H/16
        d4 = d4 + e3  # Additive Skip

        # Block 3
        d3 = self.decoder3(d4)  # -> 128, H/8
        d3 = d3 + e2  # Additive Skip

        # Block 2
        d2 = self.decoder2(d3)  # -> 64, H/4
        d2 = d2 + e1  # Additive Skip

        # Block 1
        d1 = self.decoder1(d2)  # -> 64, H/2
        d1 = d1 + e0  # Additive Skip

        # Block 0 (Final Upsample)
        d0 = self.decoder0(d1)  # -> 32, H

        # --- Head ---
        logits = self.final_conv(d0)  # -> 1, H

        return logits
