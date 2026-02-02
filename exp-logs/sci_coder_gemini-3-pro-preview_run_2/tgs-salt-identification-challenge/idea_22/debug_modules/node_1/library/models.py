import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library import config

# =============================================================================
# Building Blocks
# =============================================================================


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Additive Skip Connections.
    Structure: 1x1 Conv (Reduce) -> 3x3 Deconv (Upsample) -> 1x1 Conv (Expand).
    Internal width is set to in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Internal width calculation
        mid_channels = max(in_channels // 4, 16)

        self.block = nn.Sequential(
            # 1x1 Conv: Reduce channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 3x3 Transposed Conv: Upsample
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
            # 1x1 Conv: Expand channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        out = self.block(x)
        if skip is not None:
            # Additive Skip Connection
            out = out + skip
        return out


class FinalBlock(nn.Module):
    """
    Final upsampling block to reach original image resolution.
    """

    def __init__(self, in_channels, out_channels):
        super(FinalBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AuxDepthHead(nn.Module):
    """
    Auxiliary Regression Head for Depth Prediction.
    """

    def __init__(self, in_channels):
        super(AuxDepthHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# Base Architecture
# =============================================================================


class BaseResNet34(nn.Module):
    """
    ResNet34 Encoder modified for 1-channel input.
    """

    def __init__(self):
        super(BaseResNet34, self).__init__()
        # Load pretrained weights
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first layer: 3 channels -> 1 channel
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Sum RGB weights to initialize grayscale weights
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.sum(
                dim=1, keepdim=True
            )

        # Define encoder stages
        self.encoder0 = nn.Sequential(
            self.resnet.conv1, self.resnet.bn1, self.resnet.relu
        )
        self.encoder1 = self.resnet.maxpool
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

    def forward_encoder(self, x):
        # Assuming input is 128x128 (padded)
        e0 = self.encoder0(x)  # 64x64, 64 ch
        e1 = self.encoder1(e0)  # 32x32, 64 ch
        e2 = self.encoder2(e1)  # 32x32, 64 ch (Layer1)
        e3 = self.encoder3(e2)  # 16x16, 128 ch (Layer2)
        e4 = self.encoder4(e3)  # 8x8, 256 ch (Layer3)
        e5 = self.encoder5(e4)  # 4x4, 512 ch (Layer4/Bottleneck)

        # Return features for skips: [64x64, 32x32, 16x16, 8x8, 4x4]
        return [e0, e2, e3, e4, e5]


# =============================================================================
# Models
# =============================================================================


class PrivilegedTeacher(BaseResNet34):
    """
    Teacher model that uses Ground Truth Depth (z) as privileged information.
    Depth is injected at the bottleneck.
    """

    def __init__(self):
        super(PrivilegedTeacher, self).__init__()

        # Depth Projection MLP
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Decoder
        # Bottleneck input: e5 (512) + depth (32) = 544
        self.decoder1 = DecoderBlock(544, 256)  # Output 256 to match e4
        self.decoder2 = DecoderBlock(256, 128)  # Output 128 to match e3
        self.decoder3 = DecoderBlock(128, 64)  # Output 64 to match e2
        self.decoder4 = DecoderBlock(64, 64)  # Output 64 to match e0

        self.final = FinalBlock(64, 1)

    def forward(self, x, z):
        # Encoder
        features = self.forward_encoder(x)
        e0, e2, e3, e4, e5 = features

        # Depth Injection
        if z.dim() == 1:
            z = z.unsqueeze(1)  # (B, 1)

        z_emb = self.depth_mlp(z)  # (B, 32)
        # Expand spatially to match bottleneck
        z_emb = z_emb.unsqueeze(2).unsqueeze(3)
        z_emb = z_emb.expand(-1, -1, e5.size(2), e5.size(3))  # (B, 32, 4, 4)

        # Concatenate
        bottleneck = torch.cat([e5, z_emb], dim=1)  # (B, 544, 4, 4)

        # Decoder with skips
        d1 = self.decoder1(bottleneck, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e0)

        out = self.final(d4)
        return out


class MultiTaskStudent(BaseResNet34):
    """
    Student model that learns from image only.
    Includes an auxiliary head to predict depth, enforcing feature learning.
    """

    def __init__(self):
        super(MultiTaskStudent, self).__init__()

        # Auxiliary Depth Head
        self.aux_head = AuxDepthHead(512)

        # Decoder (Standard input sizes)
        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        self.decoder4 = DecoderBlock(64, 64)

        self.final = FinalBlock(64, 1)

    def forward(self, x):
        # Encoder
        features = self.forward_encoder(x)
        e0, e2, e3, e4, e5 = features

        # Aux Task: Predict Depth
        depth_pred = self.aux_head(e5)

        # Decoder
        d1 = self.decoder1(e5, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e0)

        mask_pred = self.final(d4)

        return mask_pred, depth_pred
