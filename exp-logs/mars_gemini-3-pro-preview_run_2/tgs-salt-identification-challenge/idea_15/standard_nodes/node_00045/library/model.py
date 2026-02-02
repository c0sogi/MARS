import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DepthProjector(nn.Module):
    """
    Non-Linear MLP to project scalar depth z into a dense embedding.
    Structure: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim=1, hidden_dim=16, output_dim=32):
        super(DepthProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        # z: (B,) or (B, 1)
        if z.dim() == 1:
            z = z.view(-1, 1)
        return self.net(z)


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Internal width is determined by in_channels // 4 to preserve information.
    Performs upsampling via Transposed Convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet logic: internal dimension based on input, not output
        mid_channels = in_channels // 4

        # 1x1 Conv to reduce channels (or project)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 3x3 Transposed Conv for upsampling
        self.trans_conv = nn.Sequential(
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
        )

        # 1x1 Conv to expand to output channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans_conv(x)
        x = self.conv2(x)
        return x


class ResNet34WideLinkNet(nn.Module):
    """
    ResNet34 Encoder + Depth Injection + Wide-LinkNet Decoder.
    """

    def __init__(self, pretrained=True):
        super(ResNet34WideLinkNet, self).__init__()

        # 1. Backbone: ResNet34
        # We load the pretrained model
        resnet = models.resnet34(pretrained=pretrained)

        # 2. Input Adaptation: Modify first layer for 1-channel input
        # Sum weights along channel dimension to preserve pretrained filters
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.sum(original_conv1.weight, dim=1, keepdim=True)
            )

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.layer1 = resnet.layer1  # 64 ch, 32x32
        self.layer2 = resnet.layer2  # 128 ch, 16x16
        self.layer3 = resnet.layer3  # 256 ch, 8x8
        self.layer4 = resnet.layer4  # 512 ch, 4x4

        # 3. Depth Injection
        self.depth_embedding_dim = 32
        self.depth_projector = DepthProjector(
            input_dim=1, hidden_dim=16, output_dim=self.depth_embedding_dim
        )

        # 4. Decoder
        # Bottleneck input: Layer4 (512) + Depth (32) = 544
        # Target output of Dec4 is 256 (to match Layer3 for addition)
        self.dec4 = DecoderBlock(512 + self.depth_embedding_dim, 256)

        # Dec3: Input 256 -> Output 128 (Match Layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: Input 128 -> Output 64 (Match Layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: Input 64 -> Output 64 (Match conv1/pool output)
        self.dec1 = DecoderBlock(64, 64)

        # Final Head: Upsample 64x64 -> 128x128
        # We use a simple transposed conv block to reach final resolution and channel count
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
            x: Image tensor (B, 1, 128, 128)
            z: Depth tensor (B,) or (B, 1)
        """
        # --- Encoder ---
        # Stem
        x0 = self.conv1(x)  # 64x64, 64
        x0_bn = self.bn1(x0)
        x0_relu = self.relu(x0_bn)
        x_pool = self.maxpool(x0_relu)  # 32x32, 64

        # Blocks
        e1 = self.layer1(x_pool)  # 32x32, 64
        e2 = self.layer2(e1)  # 16x16, 128
        e3 = self.layer3(e2)  # 8x8, 256
        e4 = self.layer4(e3)  # 4x4, 512

        # --- Depth Injection ---
        # Project depth
        d_emb = self.depth_projector(z)  # (B, 32)
        # Expand spatially to match bottleneck (4x4)
        d_emb = d_emb.unsqueeze(2).unsqueeze(3).expand(-1, -1, e4.size(2), e4.size(3))
        # Concatenate
        bottleneck = torch.cat([e4, d_emb], dim=1)  # (B, 544, 4, 4)

        # --- Decoder ---
        # Block 4: 4x4 -> 8x8
        d4 = self.dec4(bottleneck)
        d4 = d4 + e3  # Additive Skip Connection

        # Block 3: 8x8 -> 16x16
        d3 = self.dec3(d4)
        d3 = d3 + e2

        # Block 2: 16x16 -> 32x32
        d2 = self.dec2(d3)
        d2 = d2 + e1

        # Block 1: 32x32 -> 64x64
        d1 = self.dec1(d2)
        # Note: We add x0_relu (output of stem before maxpool) which is 64x64
        # But d1 output is 64x64.
        d1 = d1 + x0_relu

        # --- Final Head ---
        # 64x64 -> 128x128
        out = self.final_head(d1)

        return out
