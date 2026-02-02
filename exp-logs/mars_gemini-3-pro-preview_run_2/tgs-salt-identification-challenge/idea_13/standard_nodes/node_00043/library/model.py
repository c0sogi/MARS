import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DepthProjector(nn.Module):
    """
    Projects scalar depth to a dense embedding.
    Cite solution_lesson_node_00041: Prefer explicit injection (concatenation) over residual.
    Cite solution_lesson_node_00009: Keep embeddings small and concatenate.
    """

    def __init__(self, embedding_dim=32):
        super().__init__()
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, z):
        return self.depth_mlp(z)


class ResidualDepthInjector(nn.Module):
    """
    Injects depth information into the feature map via a residual connection.
    Projects depth embedding to match input channels and adds it.
    """

    def __init__(self, in_channels, embedding_dim=32):
        super().__init__()
        self.projector = DepthProjector(embedding_dim)
        self.adapter = nn.Linear(embedding_dim, in_channels)

    def forward(self, x, z):
        # x: (B, C, H, W)
        # z: (B, 1)
        emb = self.projector(z)  # (B, embedding_dim)
        feat = self.adapter(emb)  # (B, in_channels)

        # Broadcast (B, C) -> (B, C, 1, 1) and add to x
        return x + feat.unsqueeze(-1).unsqueeze(-1)


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Upsamples features and prepares them for additive skip connection.
    Internal width is calculated as in_channels // 4 (Wide variant).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Wide-LinkNet strategy: Use input channels to determine internal width
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv: Compress
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv: Upsample (Stride 2)
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
            # 1x1 Conv: Expand to output channels
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SaltNet(nn.Module):
    """
    Residual-Injection Wide-LinkNet with ResNet34 Backbone.
    """

    def __init__(self):
        super().__init__()

        # Load Pretrained ResNet34
        # Using default weights (ImageNet)
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # =====================================================================
        # Encoder (ResNet34)
        # =====================================================================

        # 1. Adapt First Layer (3 channels -> 1 channel)
        # We sum the weights of the RGB channels to preserve learned filters
        original_weights = base_model.conv1.weight.data  # (64, 3, 7, 7)
        new_weights = torch.sum(original_weights, dim=1, keepdim=True)  # (64, 1, 7, 7)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = new_weights

        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # ResNet Layers
        self.layer1 = base_model.layer1  # 64 ch, stride 4 (relative to input)
        self.layer2 = base_model.layer2  # 128 ch, stride 8
        self.layer3 = base_model.layer3  # 256 ch, stride 16
        self.layer4 = base_model.layer4  # 512 ch, stride 32

        # =====================================================================
        # Bottleneck (Residual Injection)
        # =====================================================================
        self.injector = ResidualDepthInjector(
            in_channels=512, embedding_dim=Config.DEPTH_EMBEDDING_DIM
        )

        # =====================================================================
        # Decoder (Wide-LinkNet)
        # =====================================================================
        # Dec4: 512 -> 256 (Matches Layer3)
        self.dec4 = DecoderBlock(512, 256)

        # Dec3: 256 -> 128 (Matches Layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: 128 -> 64 (Matches Layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: 64 -> 64 (Matches Conv1 output)
        self.dec1 = DecoderBlock(64, 64)

        # =====================================================================
        # Final Head
        # =====================================================================
        # Upsample from 64x64 (Dec1 output) to 128x128 (Original Input)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Logits
        )

    def forward(self, x, z):
        # =====================================================================
        # Encoder Pass
        # =====================================================================
        # Input: (B, 1, 128, 128)

        x0 = self.conv1(x)  # (B, 64, 64, 64)
        x0_bn = self.bn1(x0)
        x0_relu = self.relu(x0_bn)

        x_pool = self.maxpool(x0_relu)  # (B, 64, 32, 32)

        e1 = self.layer1(x_pool)  # (B, 64, 32, 32)
        e2 = self.layer2(e1)  # (B, 128, 16, 16)
        e3 = self.layer3(e2)  # (B, 256, 8, 8)
        e4 = self.layer4(e3)  # (B, 512, 4, 4)

        # =====================================================================
        # Bottleneck Injection
        # =====================================================================
        b = self.injector(e4, z)  # (B, 512, 4, 4)

        # =====================================================================
        # Decoder Pass (LinkNet Style: Additive Skips)
        # =====================================================================

        # Block 4
        d4 = self.dec4(b)  # (B, 256, 8, 8)
        d4 = d4 + e3  # Add Skip Layer 3

        # Block 3
        d3 = self.dec3(d4)  # (B, 128, 16, 16)
        d3 = d3 + e2  # Add Skip Layer 2

        # Block 2
        d2 = self.dec2(d3)  # (B, 64, 32, 32)
        d2 = d2 + e1  # Add Skip Layer 1

        # Block 1
        d1 = self.dec1(d2)  # (B, 64, 64, 64)
        d1 = d1 + x0_relu  # Add Skip Conv1 Output (before pooling)

        # =====================================================================
        # Head
        # =====================================================================
        out = self.final_up(d1)  # (B, 1, 128, 128)

        return out
