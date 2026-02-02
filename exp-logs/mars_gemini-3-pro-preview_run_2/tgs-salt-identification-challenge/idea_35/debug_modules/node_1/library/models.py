import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Additive Skip Connections.
    Structure:
    1. 1x1 Conv (Reduce channels)
    2. 3x3 Transpose Conv (Upsample)
    3. 1x1 Conv (Expand channels)
    4. Add Skip Connection
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Internal width is in_channels // 4 as per strategy
        mid_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Stride 2 upsampling
        self.deconv = nn.ConvTranspose2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)

        if skip is not None:
            out = out + skip

        return out


class SaltNet(nn.Module):
    """
    ResNet34-based Wide-LinkNet for Salt Segmentation.
    Supports two modes:
    1. Specialist Teacher: Injects depth information at the bottleneck.
    2. Generalist Student: Predicts auxiliary depth from the bottleneck.
    """

    def __init__(self, use_depth=False, aux_head=False, pretrained=True):
        super(SaltNet, self).__init__()
        self.use_depth = use_depth
        self.aux_head = aux_head

        # Load Backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Modify first conv to accept 1 channel (Grayscale)
        # Sum the weights across the channel dimension to preserve intensity info
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder Layers
        self.layer1 = resnet.layer1  # 64 ch, 1/4 res
        self.layer2 = resnet.layer2  # 128 ch, 1/8 res
        self.layer3 = resnet.layer3  # 256 ch, 1/16 res
        self.layer4 = resnet.layer4  # 512 ch, 1/32 res

        # Bottleneck Configuration
        bottleneck_channels = 512

        # Depth Injection Module (Teacher)
        if self.use_depth:
            self.depth_embed_dim = 64
            self.depth_injector = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.depth_embed_dim),
            )
            # Input to decoder will be larger due to concatenation
            bottleneck_channels += self.depth_embed_dim

        # Auxiliary Depth Head (Student)
        if self.aux_head:
            self.aux_regressor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )

        # Decoder Layers (Wide-LinkNet)
        # Dec4: 1/32 -> 1/16 (Match Layer3: 256)
        self.dec4 = DecoderBlock(bottleneck_channels, 256)

        # Dec3: 1/16 -> 1/8 (Match Layer2: 128)
        self.dec3 = DecoderBlock(256, 128)

        # Dec2: 1/8 -> 1/4 (Match Layer1: 64)
        self.dec2 = DecoderBlock(128, 64)

        # Dec1: 1/4 -> 1/2 (Match conv1 output: 64)
        # Note: Layer1 output is 64ch, Conv1 output is 64ch.
        # We upsample 32x32 (L1) -> 64x64 (Conv1).
        self.dec1 = DecoderBlock(64, 64)

        # Final Upsample: 1/2 -> 1/1 (64x64 -> 128x128)
        self.final_dec = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Logits
        )

    def forward(self, x, depth=None):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        e0 = x  # 64ch, 1/2 res (64x64)

        x = self.maxpool(x)
        e1 = self.layer1(x)  # 64ch, 1/4 res (32x32)
        e2 = self.layer2(e1)  # 128ch, 1/8 res (16x16)
        e3 = self.layer3(e2)  # 256ch, 1/16 res (8x8)
        e4 = self.layer4(e3)  # 512ch, 1/32 res (4x4)

        # Bottleneck Processing
        aux_val = None

        if self.use_depth:
            if depth is None:
                raise ValueError("Depth must be provided for Teacher model.")

            # Project depth: (B, 1) -> (B, 64)
            d = self.depth_injector(depth)
            # Reshape to (B, 64, 1, 1) and expand to (B, 64, H, W)
            d = d.view(d.size(0), self.depth_embed_dim, 1, 1)
            d = d.expand(-1, -1, e4.size(2), e4.size(3))

            # Concatenate
            center = torch.cat([e4, d], dim=1)
        else:
            center = e4
            if self.aux_head:
                aux_val = self.aux_regressor(e4)

        # Decoder
        d4 = self.dec4(center, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Final Prediction
        logits = self.final_dec(d1)

        if self.aux_head:
            return logits, aux_val

        return logits
