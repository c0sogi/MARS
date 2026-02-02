import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DepthInjector(nn.Module):
    """
    Projects scalar depth into a dense embedding vector using an MLP.
    Structure: Linear -> ReLU -> Linear.
    """

    def __init__(self, input_dim=1, output_dim=32):
        super(DepthInjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, z):
        # z shape: (Batch_Size, 1)
        return self.mlp(z)


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Performs upsampling and feature transformation.
    Internal width is determined by input channels (in_channels // 4) to preserve information.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet modification: Internal dimension based on input
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1. 1x1 Conv: Compress/Project
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 2. 3x3 Transpose Conv: Upsample (Stride 2)
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
            # 3. 1x1 Conv: Expand to target output
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    Salt Segmentation Model: ResNet34 Encoder + Depth Injection + Wide-LinkNet Decoder.
    """

    def __init__(self):
        super(ResNet34WideLinkNet, self).__init__()

        # ==========================================
        # 1. Encoder (ResNet34)
        # ==========================================
        # Load pretrained weights
        resnet = models.resnet34(pretrained=True)

        # Adapt first layer for 1-channel input (Grayscale)
        # Original: Conv2d(3, 64, ...)
        # New: Conv2d(1, 64, ...)
        # Strategy: Sum weights along the channel dimension to preserve intensity patterns
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.layer1 = resnet.layer1  # Output: 64 ch, 1/4 res
        self.layer2 = resnet.layer2  # Output: 128 ch, 1/8 res
        self.layer3 = resnet.layer3  # Output: 256 ch, 1/16 res
        self.layer4 = resnet.layer4  # Output: 512 ch, 1/32 res

        # ==========================================
        # 2. Depth Injection
        # ==========================================
        self.depth_injector = DepthInjector(
            input_dim=1, output_dim=Config.DEPTH_EMBED_DIM
        )

        # ==========================================
        # 3. Decoder (Wide-LinkNet)
        # ==========================================
        # Bottleneck Input Channels: 512 (ResNet) + 32 (Depth) = 544

        # Decoder 4: 544 -> 256 (Matches layer3)
        self.dec4 = DecoderBlock(512 + Config.DEPTH_EMBED_DIM, 256)

        # Decoder 3: 256 -> 128 (Matches layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Decoder 2: 128 -> 64 (Matches layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Decoder 1: 64 -> 64 (Matches conv1 output / layer0)
        self.dec1 = DecoderBlock(64, 64)

        # Decoder 0: 64 -> 32 (Final Upsample to original resolution)
        self.dec0 = DecoderBlock(64, 32)

        # Final Convolution: 32 -> 1 (Logits)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x, z):
        """
        Args:
            x (torch.Tensor): Image tensor (B, 1, H, W)
            z (torch.Tensor): Depth tensor (B, 1)
        """
        # --- Encoder Pass ---
        # Stem
        x0 = self.conv1(x)  # (B, 64, H/2, W/2)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)  # (B, 64, H/4, W/4)

        # Blocks
        e1 = self.layer1(x1)  # (B, 64, H/4, W/4)
        e2 = self.layer2(e1)  # (B, 128, H/8, W/8)
        e3 = self.layer3(e2)  # (B, 256, H/16, W/16)
        e4 = self.layer4(e3)  # (B, 512, H/32, W/32)

        # --- Bottleneck & Depth Injection ---
        # Project depth
        d_emb = self.depth_injector(z)  # (B, 32)
        # Expand depth embedding to spatial dimensions of e4
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)
        d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))  # (B, 32, H/32, W/32)

        # Concatenate
        cat_feat = torch.cat([e4, d_emb], dim=1)  # (B, 544, H/32, W/32)

        # --- Decoder Pass (with Additive Skips) ---

        # Block 4
        d4 = self.dec4(cat_feat)  # (B, 256, H/16, W/16)
        d4 = d4 + e3  # Additive Skip
        d4 = self.relu(d4)

        # Block 3
        d3 = self.dec3(d4)  # (B, 128, H/8, W/8)
        d3 = d3 + e2  # Additive Skip
        d3 = self.relu(d3)

        # Block 2
        d2 = self.dec2(d3)  # (B, 64, H/4, W/4)
        d2 = d2 + e1  # Additive Skip
        d2 = self.relu(d2)

        # Block 1
        d1 = self.dec1(d2)  # (B, 64, H/2, W/2)
        d1 = d1 + x0  # Additive Skip (from stem)
        d1 = self.relu(d1)

        # Block 0 (Final Upsample)
        d0 = self.dec0(d1)  # (B, 32, H, W)

        # --- Final Head ---
        out = self.final_conv(d0)  # (B, 1, H, W)

        return out
