import torch
import torch.nn as nn
from torchvision import models


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Projects a scalar depth value into scale (gamma) and shift (beta) parameters
    to modulate the feature map.
    """

    def __init__(self, input_dim=1, channels=512):
        super(FiLMLayer, self).__init__()
        # MLP: Linear -> ReLU -> Linear
        # We use 'channels' as the hidden dimension for sufficient capacity.
        self.fc1 = nn.Linear(input_dim, channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels, 2 * channels)

    def forward(self, x, z):
        """
        Args:
            x (torch.Tensor): Feature map of shape (B, C, H, W).
            z (torch.Tensor): Depth values of shape (B, 1) or (B,).
        """
        if z.dim() == 1:
            z = z.unsqueeze(1)

        # Project z to generate modulation parameters
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)  # Shape: (B, 2*C)

        # Split into gamma (scale) and beta (shift)
        gamma, beta = torch.chunk(out, 2, dim=1)

        # Reshape for broadcasting over spatial dimensions: (B, C, 1, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        # Apply affine transformation: F' = gamma * F + beta
        return x * gamma + beta


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Additive Skip Connections.
    Uses Transpose Convolution for upsampling.
    Internal width is calculated as in_channels // 4 to preserve information.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Wide-LinkNet: internal dimension based on in_channels
        internal_channels = in_channels // 4

        # 1x1 Conv to reduce dimensions (or project)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
        )

        # Transpose Conv for upsampling (3x3, stride 2)
        self.deconv = nn.Sequential(
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

        # 1x1 Conv to expand/project to output channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


class FiLMResNet34(nn.Module):
    """
    FiLM-Conditioned ResNet34-WideLinkNet.

    Architecture:
    1. Backbone: ResNet34 (pretrained), modified for 1-channel input.
    2. Bottleneck: No compression, FiLM layer modulates features based on depth.
    3. Decoder: Wide-LinkNet blocks with additive skip connections.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(FiLMResNet34, self).__init__()

        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=pretrained)

        # --- Input Adaptation ---
        # Modify first conv layer to accept 1-channel input by summing RGB weights
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            # Sum weights across channel dimension (dim 1)
            # old_conv.weight shape: (64, 3, 7, 7)
            # new_conv.weight shape: (64, 1, 7, 7)
            new_conv.weight.copy_(torch.sum(old_conv.weight, dim=1, keepdim=True))

        # Define encoder stages
        # Stem (Conv1 + BN + ReLU) - Output 64x64 (stride 2)
        self.stem = nn.Sequential(new_conv, resnet.bn1, resnet.relu)

        # MaxPool - Output 32x32 (stride 2)
        self.maxpool = resnet.maxpool

        # ResNet Layers
        self.layer1 = resnet.layer1  # 64 ch, 32x32
        self.layer2 = resnet.layer2  # 128 ch, 16x16
        self.layer3 = resnet.layer3  # 256 ch, 8x8
        self.layer4 = resnet.layer4  # 512 ch, 4x4

        # --- Bottleneck ---
        # FiLM Layer for depth conditioning
        self.film = FiLMLayer(input_dim=1, channels=512)

        # --- Decoder ---
        # Block 4: 512 -> 256 (Upsample 4x4 -> 8x8)
        self.dec4 = DecoderBlock(512, 256)

        # Block 3: 256 -> 128 (Upsample 8x8 -> 16x16)
        self.dec3 = DecoderBlock(256, 128)

        # Block 2: 128 -> 64 (Upsample 16x16 -> 32x32)
        self.dec2 = DecoderBlock(128, 64)

        # Block 1: 64 -> 64 (Upsample 32x32 -> 64x64)
        self.dec1 = DecoderBlock(64, 64)

        # --- Final Upsampling ---
        # Upsample 64x64 -> 128x128
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final classification
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, depth):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images (B, 1, 128, 128).
            depth (torch.Tensor): Depth values (B, 1).
        """
        # --- Encoder ---
        x_stem = self.stem(x)  # (B, 64, 64, 64)
        x_pool = self.maxpool(x_stem)  # (B, 64, 32, 32)

        e1 = self.layer1(x_pool)  # (B, 64, 32, 32)
        e2 = self.layer2(e1)  # (B, 128, 16, 16)
        e3 = self.layer3(e2)  # (B, 256, 8, 8)
        e4 = self.layer4(e3)  # (B, 512, 4, 4)

        # --- Bottleneck ---
        # Apply FiLM modulation using depth
        e4_mod = self.film(e4, depth)

        # --- Decoder ---
        # Additive skip connections from Encoder

        d4 = self.dec4(e4_mod)  # (B, 256, 8, 8)
        d4 = d4 + e3

        d3 = self.dec3(d4)  # (B, 128, 16, 16)
        d3 = d3 + e2

        d2 = self.dec2(d3)  # (B, 64, 32, 32)
        d2 = d2 + e1

        d1 = self.dec1(d2)  # (B, 64, 64, 64)
        d1 = d1 + x_stem  # Add stem features (pre-pool)

        # --- Final Output ---
        out = self.final_up(d1)  # (B, 32, 128, 128)
        logits = self.final_conv(out)  # (B, 1, 128, 128)

        return logits
