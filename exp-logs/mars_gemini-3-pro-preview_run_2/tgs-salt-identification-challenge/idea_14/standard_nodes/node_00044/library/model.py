import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Uses an internal dimension of in_channels // 4 to preserve information flow.
    Consists of:
    1. 1x1 Conv (Reduce)
    2. Transposed Conv (Upsample)
    3. 1x1 Conv (Expand)
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # "Calculate the internal dimension of decoder blocks as in_channels // 4"
        # This is the "Wide" modification to standard LinkNet which often uses out_channels // 4
        inter_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv to reduce channels
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            # Transposed Conv to upsample (Stride 2)
            nn.ConvTranspose2d(
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv to expand/match output channels
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WideLinkNetResNet34(nn.Module):
    """
    ResNet34-WideLinkNet with Bottleneck Concatenation and Depth Injection.
    """

    def __init__(self):
        super(WideLinkNetResNet34, self).__init__()

        # --- Backbone: ResNet34 ---
        weights = ResNet34_Weights.IMAGENET1K_V1
        self.resnet = resnet34(weights=weights)

        # --- Input Adaptation ---
        # Modify first conv layer to accept 1-channel input by summing RGB weights
        original_weights = self.resnet.conv1.weight.data
        # Summing dim 1 (channels): [64, 3, 7, 7] -> [64, 1, 7, 7]
        new_weights = original_weights.sum(dim=1, keepdim=True)

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.conv1.weight.data = new_weights

        # Extract Encoder Layers
        self.encoder0 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )
        self.maxpool = self.resnet.maxpool
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        # --- Bottleneck Depth Injection ---
        # Project scalar depth z -> Non-Linear MLP -> 32-channel embedding
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(inplace=True), nn.Linear(32, 32)
        )

        # Bottleneck calculation:
        # ResNet34 Layer 4 output: 512 channels
        # Depth Embedding: 32 channels
        # Total Bottleneck Input: 544 channels

        # --- Decoder: Wide-LinkNet ---
        # Decoder 4: 544 -> 256 (matches encoder3 output)
        self.decoder4 = DecoderBlock(512 + 32, 256)

        # Decoder 3: 256 -> 128 (matches encoder2 output)
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: 128 -> 64 (matches encoder1 output)
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: 64 -> 64 (matches encoder0 output)
        self.decoder1 = DecoderBlock(64, 64)

        # Final Upsample: 64 -> 32 (Upsample to original resolution 128x128)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final Convolution: 32 -> 1 (Logits)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, depth):
        """
        Args:
            x (torch.Tensor): Image input (B, 1, 128, 128)
            depth (torch.Tensor): Normalized depth input (B, 1)
        """
        # Ensure depth has correct shape
        if depth.dim() == 1:
            depth = depth.unsqueeze(1)

        # --- Encoder Pass ---
        # Input: 128x128
        e0 = self.encoder0(x)  # (B, 64, 64, 64)
        e0_pool = self.maxpool(e0)  # (B, 64, 32, 32)
        e1 = self.encoder1(e0_pool)  # (B, 64, 32, 32)
        e2 = self.encoder2(e1)  # (B, 128, 16, 16)
        e3 = self.encoder3(e2)  # (B, 256, 8, 8)
        e4 = self.encoder4(e3)  # (B, 512, 4, 4)

        # --- Bottleneck ---
        # Process depth
        d = self.depth_mlp(depth)  # (B, 32)
        # Spatially expand depth embedding to match feature map size (4x4)
        d = d.unsqueeze(2).unsqueeze(3)
        d = d.expand(-1, -1, e4.size(2), e4.size(3))  # (B, 32, 4, 4)

        # Concatenate features and depth
        center = torch.cat([e4, d], dim=1)  # (B, 544, 4, 4)

        # --- Decoder Pass (with Additive Skip Connections) ---
        d4 = self.decoder4(center)  # (B, 256, 8, 8)
        d4 = d4 + e3  # Additive Skip

        d3 = self.decoder3(d4)  # (B, 128, 16, 16)
        d3 = d3 + e2  # Additive Skip

        d2 = self.decoder2(d3)  # (B, 64, 32, 32)
        d2 = d2 + e1  # Additive Skip

        d1 = self.decoder1(d2)  # (B, 64, 64, 64)
        d1 = d1 + e0  # Additive Skip

        # --- Final Projection ---
        out = self.final_upsample(d1)  # (B, 32, 128, 128)
        out = self.final_conv(out)  # (B, 1, 128, 128)

        return out
