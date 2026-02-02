import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Module.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel SE (cSE)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial SE (sSE)
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent excitation: Channel attention + Spatial attention
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    LinkNet-style Decoder Block with scSE Attention.
    Performs Upsampling -> Projection -> Addition (Skip) -> Attention -> Convolution.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Upsample input to match spatial dimensions of the skip connection
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Project upsampled features to match skip_channels for additive connection
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, skip_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
        )

        # Attention on the combined features
        self.scse = SCSEModule(skip_channels)

        # Process features to desired output channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        x = self.conv1(x)

        if skip is not None:
            # LinkNet uses additive skip connections
            x = x + skip

        x = self.scse(x)
        x = self.conv2(x)
        return x


class SaltLinkNet(nn.Module):
    """
    Depth-Conditioned Attention-LinkNet with SE-ResNeXt50 Backbone.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder Setup
        self.encoder = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Adapt first conv for 1-channel input (Grayscale)
        if hasattr(self.encoder, "conv1"):
            old_conv = self.encoder.conv1
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize by summing weights across RGB channels
            with torch.no_grad():
                new_conv.weight[:] = torch.sum(old_conv.weight, dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
            self.encoder.conv1 = new_conv

        # Extract channel counts for skip connections
        # e.g., [64, 256, 512, 1024, 2048] for ResNeXt50
        c0, c1, c2, c3, c4 = self.encoder.feature_info.channels()

        # 2. Depth Embedding
        self.depth_emb_dim = Config.DEPTH_EMBEDDING_DIM
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, self.depth_emb_dim), nn.ReLU(inplace=True)
        )

        # 3. Decoder Pathway
        # Bottleneck Input: Encoder Stage 4 (c4) + Depth Embedding
        # Block 4: Processes bottleneck, adds Skip Stage 3 (c3)
        self.dec4 = DecoderBlock(c4 + self.depth_emb_dim, c3, 256)

        # Block 3: Processes dec4 out, adds Skip Stage 2 (c2)
        self.dec3 = DecoderBlock(256, c2, 128)

        # Block 2: Processes dec3 out, adds Skip Stage 1 (c1)
        self.dec2 = DecoderBlock(128, c1, 64)

        # Block 1: Processes dec2 out, adds Skip Stage 0 (c0)
        self.dec1 = DecoderBlock(64, c0, 32)

        # 4. Final Head
        # Upsample from stride 2 to stride 1 (Original Resolution)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x, depth):
        """
        Args:
            x (torch.Tensor): Image input (B, 1, H, W)
            depth (torch.Tensor): Depth scalar (B, 1)
        """
        # Encoder Forward
        # e0: stride 2, e1: stride 4, e2: stride 8, e3: stride 16, e4: stride 32
        e0, e1, e2, e3, e4 = self.encoder(x)

        # Depth Injection
        # Project depth scalar to embedding
        d = self.depth_mlp(depth)  # (B, 16)
        # Expand spatially to match bottleneck features (H/32, W/32)
        d = d.unsqueeze(2).unsqueeze(3)
        d = d.expand(-1, -1, e4.size(2), e4.size(3))

        # Concatenate at bottleneck
        bottleneck = torch.cat([e4, d], dim=1)

        # Decoder Forward
        d4 = self.dec4(bottleneck, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Final Upsample and Prediction
        out = self.final_up(d1)
        logits = self.final_conv(out)

        return logits
