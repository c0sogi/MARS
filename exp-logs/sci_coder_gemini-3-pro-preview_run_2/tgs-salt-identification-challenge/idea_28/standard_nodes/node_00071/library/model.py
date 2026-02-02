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


class AuxDepthHead(nn.Module):
    """
    Auxiliary Depth Regression Head.

    Attached to the bottleneck feature map to predict the normalized depth of the image.
    This forces the encoder to learn depth-correlated texture features.

    Structure: GlobalAveragePooling -> MLP -> Scalar.
    """

    def __init__(self, in_channels):
        super(AuxDepthHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class SaltNet(nn.Module):
    """
    Corrected Multi-Task Wide-LinkNet Architecture.

    Backbone: ResNet34 (Pretrained).
    Decoder: Wide-LinkNet with Additive Skip Connections.
    Auxiliary Task: Depth Regression (Not injected into decoder).
    """

    def __init__(self):
        super(SaltNet, self).__init__()

        # Load Pretrained ResNet34
        weights = ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)

        # ---------------------------------------------------------------------
        # Encoder (ResNet34)
        # ---------------------------------------------------------------------

        # Input Adaptation: Modify first conv to accept 1-channel input
        # We sum the weights of the original 3 channels to preserve feature detectors
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
        # Bottleneck & Aux Head
        # ---------------------------------------------------------------------

        # No explicit compression layer; pass full 512 channels to decoder.
        self.aux_head = AuxDepthHead(512)

        # ---------------------------------------------------------------------
        # Decoder (Wide-LinkNet)
        # ---------------------------------------------------------------------

        # Decoder 4: 512 -> 256. Skip: layer3 (256)
        self.decoder4 = DecoderBlock(512, 256)

        # Decoder 3: 256 -> 128. Skip: layer2 (128)
        self.decoder3 = DecoderBlock(256, 128)

        # Decoder 2: 128 -> 64. Skip: layer1 (64)
        self.decoder2 = DecoderBlock(128, 64)

        # Decoder 1: 64 -> 64. Skip: conv1 output (64)
        # Note: layer1 output (input to d2) has same spatial dim as maxpool output.
        # conv1 output (skip for d1) is 2x larger (Stride 2).
        self.decoder1 = DecoderBlock(64, 64)

        # ---------------------------------------------------------------------
        # Final Head
        # ---------------------------------------------------------------------

        # Upsample from Stride 2 (64x64) to Stride 1 (128x128)
        # Then map to 1 output channel (logits)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # ---------------------------------------------------------------------
        # Encoder Path
        # ---------------------------------------------------------------------
        # x: (B, 1, H, W)

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
        # Aux Task
        # ---------------------------------------------------------------------
        # Predict depth from bottleneck features
        depth_pred = self.aux_head(e4)

        # ---------------------------------------------------------------------
        # Decoder Path
        # ---------------------------------------------------------------------
        # Depth is NOT injected into decoder to ensure test-time compatibility

        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        # ---------------------------------------------------------------------
        # Final Prediction
        # ---------------------------------------------------------------------
        mask_logits = self.final(d1)

        return mask_logits, depth_pred
