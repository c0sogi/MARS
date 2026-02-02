import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DecoderBlock(nn.Module):
    """
    LinkNet-style decoder block with modified internal width.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet: Calculate internal dimension as in_channels // 4
        # This creates a wider bottleneck than standard LinkNet (out // 4)
        # to prevent information bottlenecks.
        internal_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1x1 Conv: Reduce dimensions
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
            # 1x1 Conv: Expand to output dimensions
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34WideLinkNet(nn.Module):
    """
    Corrected Multi-Task Wide-LinkNet with ResNet34 Backbone.
    """

    def __init__(self, pretrained=True):
        super(ResNet34WideLinkNet, self).__init__()

        # Load backbone
        # We use pretrained=True for compatibility.
        backbone = resnet34(pretrained=pretrained)

        # --- Input Adaptation ---
        # Modify the first convolutional layer to accept 1-channel input (Grayscale)
        # We sum the weights of the original RGB channels to preserve filter structure.
        original_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.sum(original_conv1.weight, dim=1, keepdim=True)
            )

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Encoder Layers
        self.layer1 = backbone.layer1  # Output: 64 ch
        self.layer2 = backbone.layer2  # Output: 128 ch
        self.layer3 = backbone.layer3  # Output: 256 ch
        self.layer4 = backbone.layer4  # Output: 512 ch (Bottleneck)

        # --- Auxiliary Depth Head ---
        # Attached to the bottleneck (Layer 4 output)
        # GlobalAveragePooling -> MLP -> Scalar Depth Prediction
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
        )

        # --- Decoder ---
        # Wide-LinkNet blocks with Additive Skip Connections

        # Block 4: 512 -> 256 (Matches Layer3)
        self.dec4 = DecoderBlock(512, 256)

        # Block 3: 256 -> 128 (Matches Layer2)
        self.dec3 = DecoderBlock(256, 128)

        # Block 2: 128 -> 64 (Matches Layer1)
        self.dec2 = DecoderBlock(128, 64)

        # Block 1: 64 -> 64 (Matches Initial Conv output)
        self.dec1 = DecoderBlock(64, 64)

        # --- Final Head ---
        # Upsample from 1/2 resolution to Full resolution
        self.final_trans = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Binary Segmentation Logits
        )

    def forward(self, x):
        # --- Encoder Path ---
        # x: (B, 1, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)  # (B, 64, H/2, W/2) -> Skip for Dec1

        x = self.maxpool(c1)  # (B, 64, H/4, W/4)
        e1 = self.layer1(x)  # (B, 64, H/4, W/4) -> Skip for Dec2
        e2 = self.layer2(e1)  # (B, 128, H/8, W/8) -> Skip for Dec3
        e3 = self.layer3(e2)  # (B, 256, H/16, W/16) -> Skip for Dec4
        e4 = self.layer4(e3)  # (B, 512, H/32, W/32) -> Bottleneck

        # --- Auxiliary Task ---
        # Predict depth from bottleneck features
        depth_pred = self.aux_head(e4)

        # --- Decoder Path ---
        # Block 4
        d4 = self.dec4(e4)  # Upsample to H/16
        d4 = d4 + e3  # Additive Skip

        # Block 3
        d3 = self.dec3(d4)  # Upsample to H/8
        d3 = d3 + e2  # Additive Skip

        # Block 2
        d2 = self.dec2(d3)  # Upsample to H/4
        d2 = d2 + e1  # Additive Skip

        # Block 1
        d1 = self.dec1(d2)  # Upsample to H/2
        d1 = d1 + c1  # Additive Skip

        # Final Upsample
        out = self.final_trans(d1)  # Upsample to H
        mask_logits = self.final_conv(out)

        return mask_logits, depth_pred
