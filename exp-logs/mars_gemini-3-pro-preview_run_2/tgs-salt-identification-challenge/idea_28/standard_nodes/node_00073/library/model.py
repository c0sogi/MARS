import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet34_Weights


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.

    Implements the LinkNet decoder block with a 'Wide' modification where the
    internal dimension is calculated as in_channels // 4 (preserving more information)
    rather than the standard out_channels // 4.

    Structure:
        1x1 Conv (Reduce) -> 3x3 Transpose Conv (Upsample) -> 1x1 Conv (Expand) -> Add Skip
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet: Calculate internal dimension as in_channels // 4
        internal_channels = in_channels // 4

        # 1. Reduction (1x1 Conv)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Upsampling (3x3 Transpose Conv, Stride 2)
        self.trans = nn.Sequential(
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

        # 3. Expansion (1x1 Conv)
        self.conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        """
        Args:
            x (torch.Tensor): Input tensor from the deeper layer (N, in_channels, H, W).
            skip (torch.Tensor): Skip connection tensor from the encoder (N, out_channels, 2H, 2W).

        Returns:
            torch.Tensor: Combined output (N, out_channels, 2H, 2W).
        """
        x = self.conv1(x)
        x = self.trans(x)
        x = self.conv2(x)

        # Additive Skip Connection
        return x + skip


class SaltNet(nn.Module):
    """
    Wide-LinkNet Architecture with Explicit Depth Injection.

    Backbone: ResNet34 (Pretrained).
    Decoder: Wide-LinkNet with Additive Skip Connections.
    Depth Injection: Concatenation at Bottleneck (Cite solution_lesson_node_00024, solution_lesson_node_00037).
    """

    def __init__(self):
        super(SaltNet, self).__init__()

        # Load Pretrained ResNet34
        weights = ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)

        # ---------------------------------------------------------------------
        # Encoder (ResNet34)
        # ---------------------------------------------------------------------

        # Input Adaptation: Modify first conv to accept 1-channel input (Cite solution_lesson_node_00028)
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:] = original_conv1.weight.sum(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels, Stride 4 (effective)
        self.layer2 = resnet.layer2  # 128 channels, Stride 8
        self.layer3 = resnet.layer3  # 256 channels, Stride 16
        self.layer4 = resnet.layer4  # 512 channels, Stride 32

        # ---------------------------------------------------------------------
        # Depth Injection Module
        # ---------------------------------------------------------------------
        # MLP to project scalar depth to embedding (Cite solution_lesson_node_00029)
        self.depth_projector = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 32)
        )

        # Bottleneck channels: 512 (ResNet) + 32 (Depth) = 544

        # ---------------------------------------------------------------------
        # Decoder (Wide-LinkNet)
        # ---------------------------------------------------------------------

        # Decoder 4: 544 -> 256. Skip: layer3 (256)
        # Internal width will be 544 // 4 = 136 (Cite solution_lesson_node_00023)
        self.decoder4 = DecoderBlock(544, 256)

        # Decoder 3: 256 -> 128. Skip: layer2 (128)
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: 128 -> 64. Skip: layer1 (64)
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: 64 -> 64. Skip: conv1 output (64)
        self.decoder1 = DecoderBlock(64, 64)

        # ---------------------------------------------------------------------
        # Final Head
        # ---------------------------------------------------------------------

        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, depth):
        # ---------------------------------------------------------------------
        # Encoder Path
        # ---------------------------------------------------------------------
        # x: (B, 1, H, W)
        # depth: (B, 1)

        # Stride 2
        x = self.conv1(x)
        x = self.bn1(x)
        e0 = self.relu(x)  # Save for Decoder 1 (64 ch)

        # Stride 4
        x = self.maxpool(e0)
        e1 = self.layer1(x)  # Save for Decoder 2 (64 ch)

        # Stride 8
        e2 = self.layer2(e1)  # Save for Decoder 3 (128 ch)

        # Stride 16
        e3 = self.layer3(e2)  # Save for Decoder 4 (256 ch)

        # Stride 32
        e4 = self.layer4(e3)  # Bottleneck (512 ch)

        # ---------------------------------------------------------------------
        # Depth Injection
        # ---------------------------------------------------------------------
        # Project depth
        d_emb = self.depth_projector(depth)  # (B, 32)
        d_emb = d_emb.unsqueeze(2).unsqueeze(3)  # (B, 32, 1, 1)

        # Tile to match feature map spatial dims (H/32, W/32)
        # e4 is (B, 512, 4, 4) for 128x128 input
        d_emb = d_emb.expand(-1, -1, e4.size(2), e4.size(3))

        # Concatenate (Cite solution_lesson_node_00037, solution_lesson_node_00041)
        e4_cat = torch.cat([e4, d_emb], dim=1)  # (B, 544, 4, 4)

        # ---------------------------------------------------------------------
        # Decoder Path
        # ---------------------------------------------------------------------

        d4 = self.decoder4(e4_cat, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        # ---------------------------------------------------------------------
        # Final Prediction
        # ---------------------------------------------------------------------
        mask_logits = self.final(d1)

        return mask_logits
