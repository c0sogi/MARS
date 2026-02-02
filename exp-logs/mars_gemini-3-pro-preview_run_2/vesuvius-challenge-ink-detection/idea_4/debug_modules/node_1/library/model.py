import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    Includes logic to handle slight shape mismatches during concatenation.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # Bilinear upsampling to double the spatial dimensions
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Handle potential shape mismatch (e.g., due to odd dimensions in encoder pooling)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class HPUnet(nn.Module):
    """
    Hybrid-Projection U-Net (HPUnet) with ResNet34 Encoder.

    This model is designed for the Vesuvius Ink Detection task. It uses a ResNet34 backbone
    pretrained on ImageNet. The first layer is modified to accept 4 input channels to
    accommodate the Hybrid-Projection input (Global MIP + 3 Stratified MIPs).
    """

    def __init__(self, in_channels=4, classes=1):
        super().__init__()

        # Load ResNet34 with modern ImageNet weights
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        self.encoder = models.resnet34(weights=weights)

        # --- Modify Input Layer ---
        # Original ResNet34 conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.encoder.conv1

        # Create a new conv1 layer with 'in_channels' (4) instead of 3
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize weights for the new layer
        with torch.no_grad():
            # Copy weights for the first 3 channels (RGB equivalent)
            self.encoder.conv1.weight[:, :3] = original_conv1.weight

            # For the extra channels (e.g., the 4th channel), initialize with the mean of RGB weights.
            # This preserves the edge/texture detection filters of the pre-trained model while
            # adapting to the new modality.
            if in_channels > 3:
                # Calculate mean across RGB channels: shape (64, 1, 7, 7)
                mean_weight = original_conv1.weight.mean(dim=1, keepdim=True)
                # Repeat for the remaining channels
                self.encoder.conv1.weight[:, 3:] = mean_weight.repeat(
                    1, in_channels - 3, 1, 1
                )

        # --- Decoder Construction ---
        # ResNet34 channel sizes: layer1=64, layer2=128, layer3=256, layer4=512

        # Decoder 4: Processes bottleneck (512) and skips from layer3 (256)
        self.decoder4 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )

        # Decoder 3: Processes output of dec4 (256) and skips from layer2 (128)
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Decoder 2: Processes output of dec3 (128) and skips from layer1 (64)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Decoder 1: Processes output of dec2 (64) and skips from initial conv1/relu (64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # --- Segmentation Head ---
        # Upsample from Decoder 1 resolution (H/2) to original resolution (H)
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        # Final convolution to reduce to 1 class (binary mask)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder Path ---
        # Input x: (Batch, 4, H, W)

        # Stage 0: Initial Conv -> BN -> ReLU
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        # x0 shape: (B, 64, H/2, W/2) -> Skip connection for Decoder 1

        # Stage 1: MaxPool -> Layer 1
        x1 = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x1)
        # x1 shape: (B, 64, H/4, W/4) -> Skip connection for Decoder 2

        # Stage 2: Layer 2
        x2 = self.encoder.layer2(x1)
        # x2 shape: (B, 128, H/8, W/8) -> Skip connection for Decoder 3

        # Stage 3: Layer 3
        x3 = self.encoder.layer3(x2)
        # x3 shape: (B, 256, H/16, W/16) -> Skip connection for Decoder 4

        # Stage 4: Layer 4 (Bottleneck)
        x4 = self.encoder.layer4(x3)
        # x4 shape: (B, 512, H/32, W/32)

        # --- Decoder Path ---
        d4 = self.decoder4(x4, x3)  # Output: (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, x2)  # Output: (B, 128, H/8, W/8)
        d2 = self.decoder2(d3, x1)  # Output: (B, 64, H/4, W/4)
        d1 = self.decoder1(d2, x0)  # Output: (B, 64, H/2, W/2)

        # --- Head ---
        out = self.final_upsample(d1)  # Output: (B, 64, H, W)
        out = self.final_conv(out)  # Output: (B, 1, H, W)

        return out
