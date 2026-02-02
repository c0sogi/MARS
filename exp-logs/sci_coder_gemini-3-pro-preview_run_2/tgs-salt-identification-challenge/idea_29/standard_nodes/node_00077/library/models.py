import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block.
    Consists of:
    1. 1x1 Conv (reduce/expand)
    2. 3x3 Transpose Conv (upsample)
    3. 1x1 Conv (expand/reduce)

    The 'internal width' is set to in_channels // 4 as per the Wide-LinkNet specification.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Internal width calculation
        mid_channels = in_channels // 4

        # Ensure mid_channels is at least 1 to avoid errors with small channel counts
        if mid_channels < 1:
            mid_channels = 1

        self.block = nn.Sequential(
            # 1x1 Conv to reduce channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transpose Conv to upsample
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


class ResNet34Encoder(nn.Module):
    """
    ResNet34 Encoder backbone.
    Modified to accept 1-channel input by summing the weights of the first convolution.
    Returns features at 5 scales: x0 (H/2), x1 (H/4), x2 (H/8), x3 (H/16), x4 (H/32).
    """

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet34
        # Using 'IMAGENET1K_V1' as per standard modern torchvision usage
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first convolution for 1-channel input
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Sum the weights along the channel dimension: (64, 3, 7, 7) -> (64, 1, 7, 7)
        with torch.no_grad():
            self.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        # x: (B, 1, H, W)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)  # x0: (B, 64, H/2, W/2)

        x = self.maxpool(x0)

        # Encoder Layers
        x1 = self.layer1(x)  # x1: (B, 64, H/4, W/4)
        x2 = self.layer2(x1)  # x2: (B, 128, H/8, W/8)
        x3 = self.layer3(x2)  # x3: (B, 256, H/16, W/16)
        x4 = self.layer4(x3)  # x4: (B, 512, H/32, W/32) - Bottleneck

        return x0, x1, x2, x3, x4


class SaltModel(nn.Module):
    """
    Salt Segmentation Model (ResNet34 + Depth Injection + LinkNet).
    Cite Lesson 00032: Explicit Feature Injection is superior to distillation for orthogonal metadata.
    Cite Lesson 00037: Concatenation is superior to FiLM/Multiplication for weak signals.
    """

    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = ResNet34Encoder()

        # Depth Injection Module
        # Projects scalar depth to 64 channels
        # Cite Lesson 00029: Non-linear MLP embeddings outperform linear projections.
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # Decoder
        # Bottleneck input channels: 512 (ResNet) + 64 (Depth) = 576

        # Block 4: 576 -> 256 (Matches x3 channels for skip)
        self.decoder4 = DecoderBlock(512 + 64, 256)

        # Block 3: 256 -> 128 (Matches x2 channels for skip)
        self.decoder3 = DecoderBlock(256, 128)

        # Block 2: 128 -> 64 (Matches x1 channels for skip)
        self.decoder2 = DecoderBlock(128, 64)

        # Block 1: 64 -> 64 (Matches x0 channels for skip)
        self.decoder1 = DecoderBlock(64, 64)

        # Final Head: Upsample from H/2 to H
        self.final_head = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x, depth):
        # x: (B, 1, H, W)
        # depth: (B, 1)

        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Depth Injection
        # Project depth: (B, 1) -> (B, 64)
        d = self.depth_mlp(depth)
        # Reshape to (B, 64, 1, 1) and expand to bottleneck spatial dimensions
        d = d.unsqueeze(-1).unsqueeze(-1)
        d = d.expand(-1, -1, x4.size(2), x4.size(3))

        # Concatenate: (B, 512, H/32, W/32) + (B, 64, H/32, W/32) -> (B, 576, H/32, W/32)
        bottleneck = torch.cat([x4, d], dim=1)

        # Decoder with Additive Skip Connections
        d4 = self.decoder4(bottleneck)  # -> (B, 256, H/16, W/16)
        d4 = d4 + x3

        d3 = self.decoder3(d4)  # -> (B, 128, H/8, W/8)
        d3 = d3 + x2

        d2 = self.decoder2(d3)  # -> (B, 64, H/4, W/4)
        d2 = d2 + x1

        d1 = self.decoder1(d2)  # -> (B, 64, H/2, W/2)
        d1 = d1 + x0

        # Final Prediction
        logits = self.final_head(d1)  # -> (B, num_classes, H, W)

        return logits
