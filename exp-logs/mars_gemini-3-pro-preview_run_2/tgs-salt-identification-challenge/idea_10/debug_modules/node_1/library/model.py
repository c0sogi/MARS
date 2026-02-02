import torch
import torch.nn as nn
from torchvision import models


class DepthEmbedding(nn.Module):
    """
    Projects scalar depth into a vector embedding.
    Applies dropout to the entire embedding vector to force robustness.
    """

    def __init__(self, input_dim=1, hidden_dim=16, output_dim=32, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z):
        # z shape: (Batch_Size, 1) or (Batch_Size,)
        if z.dim() == 1:
            z = z.unsqueeze(1)

        # Project depth
        emb = self.mlp(z)

        # Apply dropout to the embedding vector
        emb = self.dropout(emb)

        # Reshape for concatenation with feature maps: (B, C, 1, 1)
        return emb.unsqueeze(2).unsqueeze(3)


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block with 'Wide' internal channels.
    Internal dimension is calculated as in_channels // 4 to preserve information.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Wide-LinkNet strategy: internal width based on input, not output
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv: Reduce dimension
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
            # 1x1 Conv: Expand/Project to target output dimension
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthRobustLinkNet(nn.Module):
    """
    ResNet34-based LinkNet with Depth Injection and Wide Decoder Blocks.
    """

    def __init__(self, in_channels=1, n_classes=1):
        super().__init__()

        # 1. Backbone: ResNet34
        # Using IMAGENET1K_V1 weights
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first convolution to accept `in_channels` (1) instead of 3
        # We sum the weights across the channel dimension to preserve pretrained filters
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            # Sum RGB weights to get 1-channel weights: (64, 3, 7, 7) -> (64, 1, 7, 7)
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))

        self.initial = nn.Sequential(new_conv, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # 2. Depth Injection Module
        self.depth_embedding = DepthEmbedding(output_dim=32)

        # 3. Decoder
        # Bottleneck: Encoder4 (512) + Depth (32) = 544 channels
        # Decoder 4: 544 -> 256 (matches Encoder3)
        self.decoder4 = DecoderBlock(544, 256)

        # Decoder 3: 256 (Dec4 + Enc3 skip) -> 128 (matches Encoder2)
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: 128 (Dec3 + Enc2 skip) -> 64 (matches Encoder1)
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: 64 (Dec2 + Enc1 skip) -> 64 (matches Initial output)
        self.decoder1 = DecoderBlock(64, 64)

        # Final Upsampling Head
        # Input: 64 (Dec1 + Initial skip). Output: n_classes.
        # Upsamples from 64x64 to 128x128 (assuming 128 input size)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, 1, H, W)
            z: Depth tensor (B, 1) or (B,)
        """
        # --- Encoder ---
        # Initial Block
        x0 = self.initial(x)  # 64 ch, stride 2
        x_mp = self.maxpool(x0)  # 64 ch, stride 4

        # ResNet Layers
        e1 = self.encoder1(x_mp)  # 64 ch, stride 4 (32x32 for 128 input)
        e2 = self.encoder2(e1)  # 128 ch, stride 8
        e3 = self.encoder3(e2)  # 256 ch, stride 16
        e4 = self.encoder4(e3)  # 512 ch, stride 32

        # --- Depth Injection ---
        d = self.depth_embedding(z)  # (B, 32, 1, 1)
        # Expand depth embedding to match spatial resolution of bottleneck
        d = d.expand(-1, -1, e4.size(2), e4.size(3))

        # Concatenate depth with bottleneck features
        bottleneck = torch.cat([e4, d], dim=1)  # 512 + 32 = 544 channels

        # --- Decoder (LinkNet Style with Additive Skips) ---
        # Block 4
        d4 = self.decoder4(bottleneck)
        d4 = d4 + e3

        # Block 3
        d3 = self.decoder3(d4)
        d3 = d3 + e2

        # Block 2
        d2 = self.decoder2(d3)
        d2 = d2 + e1

        # Block 1
        d1 = self.decoder1(d2)
        d1 = d1 + x0  # Additive skip from initial block

        # --- Final Head ---
        out = self.final_up(d1)

        return out
