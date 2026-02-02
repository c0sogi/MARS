import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class DecoderBlock(nn.Module):
    """
    LinkNet-style decoder block with width correction.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Wide-LinkNet: Internal dimension is in_channels // 4
        # This preserves more information than standard LinkNet which might compress more aggressively.
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv: Compress
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transpose Conv: Upsample
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
            # 1x1 Conv: Expand to target output
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34 Encoder + Wide LinkNet Decoder with Depth Injection.
    """

    def __init__(self):
        super().__init__()

        # Load Pretrained ResNet34
        weights = ResNet34_Weights.IMAGENET1K_V1
        backbone = resnet34(weights=weights)

        # --- Input Adaptation ---
        # Modify first layer to accept 1 channel instead of 3
        # Sum the weights across the channel dimension to preserve pretrained filters
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder Layers
        self.encoder1 = backbone.layer1  # 64 channels, 1/4 scale (same as maxpool out)
        self.encoder2 = backbone.layer2  # 128 channels, 1/8 scale
        self.encoder3 = backbone.layer3  # 256 channels, 1/16 scale
        self.encoder4 = backbone.layer4  # 512 channels, 1/32 scale

        # --- Depth Injection ---
        # MLP to project scalar depth to embedding
        self.depth_mlp = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 32))

        # --- Decoder ---
        # Decoder 4: Takes Bottleneck (Enc4 + Depth) -> Matches Enc3
        # Input: 512 + 32 = 544. Output: 256.
        self.decoder4 = DecoderBlock(544, 256)

        # Decoder 3: Takes Dec4 out -> Matches Enc2
        # Input: 256. Output: 128.
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: Takes Dec3 out -> Matches Enc1
        # Input: 128. Output: 64.
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: Takes Dec2 out -> Matches Conv1 output
        # Input: 64. Output: 64.
        self.decoder1 = DecoderBlock(64, 64)

        # Final Head: Upsample from 1/2 scale (Conv1 size) to Full Scale
        self.final_head = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, z):
        """
        Args:
            x: Input images (B, 1, H, W)
            z: Depth values (B, 1) or (B,)
        """
        # Ensure z is (B, 1)
        if z.dim() == 1:
            z = z.unsqueeze(1)

        # --- Encoder ---
        # Initial Block
        x0 = self.conv1(x)  # (B, 64, H/2, W/2)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # Save for skip connection
        x_pool = self.maxpool(x0)  # (B, 64, H/4, W/4)

        # ResNet Blocks
        e1 = self.encoder1(x_pool)  # (B, 64, H/4, W/4)
        e2 = self.encoder2(e1)  # (B, 128, H/8, W/8)
        e3 = self.encoder3(e2)  # (B, 256, H/16, W/16)
        e4 = self.encoder4(e3)  # (B, 512, H/32, W/32)

        # --- Depth Injection ---
        d_emb = self.depth_mlp(z)  # (B, 32)
        # Expand spatially to match bottleneck features
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))

        # Concatenate
        bottleneck = torch.cat([e4, d_emb], dim=1)  # (B, 544, H/32, W/32)

        # --- Decoder ---
        # Block 4
        d4 = self.decoder4(bottleneck)  # -> (B, 256, H/16, W/16)
        d4 = d4 + e3  # Additive Skip

        # Block 3
        d3 = self.decoder3(d4)  # -> (B, 128, H/8, W/8)
        d3 = d3 + e2  # Additive Skip

        # Block 2
        d2 = self.decoder2(d3)  # -> (B, 64, H/4, W/4)
        d2 = d2 + e1  # Additive Skip

        # Block 1
        d1 = self.decoder1(d2)  # -> (B, 64, H/2, W/2)
        d1 = d1 + x0  # Additive Skip (Conv1 output)

        # Final Upsample
        out = self.final_head(d1)  # -> (B, 1, H, W)

        return out
