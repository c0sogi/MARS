import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from library.config import Config


class ConvBlock3d(nn.Module):
    """
    Basic 3D Convolution Block: Conv3d -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock3d(nn.Module):
    """
    3D Decoder Block with Skip Connection.
    Upsamples input, concatenates with skip features, and refines with convolutions.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock3d, self).__init__()
        # We use interpolation for upsampling followed by a conv to reduce artifacts
        # Alternatively, ConvTranspose3d could be used.
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        # Convolution to reduce channels after concatenation
        self.conv1 = ConvBlock3d(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock3d(out_channels, out_channels)

    def forward(self, x, skip):
        # Upsample
        x = self.up(x)

        # Handle potential shape mismatch due to padding/cropping in encoder
        # x shape: (B, C, D, H, W)
        # skip shape: (B, C_skip, D_skip, H_skip, W_skip)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Convolve
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet3DUNet(nn.Module):
    """
    3D U-Net with ResNet-18 Backbone.
    """

    def __init__(
        self, in_channels=Config.IN_CHANNELS, out_channels=Config.OUT_CHANNELS
    ):
        super(ResNet3DUNet, self).__init__()

        # --- Encoder (ResNet-18 3D) ---
        # Load pre-trained weights
        weights = R3D_18_Weights.DEFAULT
        self.backbone = r3d_18(weights=weights)

        # Modify the first layer (Stem) to accept 'in_channels' (1 for MRI)
        # Original: Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        original_stem0 = self.backbone.stem[0]
        self.backbone.stem[0] = nn.Conv3d(
            in_channels,
            original_stem0.out_channels,
            kernel_size=original_stem0.kernel_size,
            stride=original_stem0.stride,
            padding=original_stem0.padding,
            bias=False,
        )

        # Initialize the new layer with averaged weights from the pre-trained model
        with torch.no_grad():
            self.backbone.stem[0].weight.data = original_stem0.weight.data.mean(
                dim=1, keepdim=True
            )

        # Encoder Layers
        self.encoder_stem = self.backbone.stem  # Out: 64 ch, (D, H/2, W/2)
        self.encoder_layer1 = self.backbone.layer1  # Out: 64 ch, (D, H/2, W/2)
        self.encoder_layer2 = self.backbone.layer2  # Out: 128 ch, (D/2, H/4, W/4)
        self.encoder_layer3 = self.backbone.layer3  # Out: 256 ch, (D/4, H/8, W/8)
        self.encoder_layer4 = self.backbone.layer4  # Out: 512 ch, (D/8, H/16, W/16)

        # --- Decoder ---
        # Layer 4 -> Layer 3
        # In: 512, Skip: 256 (Layer3), Out: 256
        self.dec4 = DecoderBlock3d(512, 256, 256)

        # Layer 3 -> Layer 2
        # In: 256, Skip: 128 (Layer2), Out: 128
        self.dec3 = DecoderBlock3d(256, 128, 128)

        # Layer 2 -> Layer 1
        # In: 128, Skip: 64 (Layer1), Out: 64
        self.dec2 = DecoderBlock3d(128, 64, 64)

        # Layer 1 -> Stem (Note: Layer1 and Stem usually have same resolution in r3d_18)
        # We can just process Layer 1 output combined with Stem if needed, or just upsample from Layer 1 to Input.
        # r3d_18 Stem stride is (1, 2, 2). Layer 1 stride is 1.
        # So Layer 1 output is (D, H/2, W/2).
        # We need to get back to (D, H, W).

        # Final Upsampling Block
        # In: 64, Out: 32
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            ConvBlock3d(64, 32),
        )

        # Final Projection
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # x: (B, 1, D, H, W)

        x_stem = self.encoder_stem(x)  # (B, 64, D, H/2, W/2)
        x_l1 = self.encoder_layer1(x_stem)  # (B, 64, D, H/2, W/2)
        x_l2 = self.encoder_layer2(x_l1)  # (B, 128, D/2, H/4, W/4)
        x_l3 = self.encoder_layer3(x_l2)  # (B, 256, D/4, H/8, W/8)
        x_l4 = self.encoder_layer4(x_l3)  # (B, 512, D/8, H/16, W/16)

        # --- Decoder ---
        d4 = self.dec4(x_l4, x_l3)  # (B, 256, D/4, H/8, W/8)
        d3 = self.dec3(d4, x_l2)  # (B, 128, D/2, H/4, W/4)
        d2 = self.dec2(d3, x_l1)  # (B, 64, D, H/2, W/2)

        # Final Upsample to original resolution
        # We are at (D, H/2, W/2), need (D, H, W)
        out = self.final_up(d2)  # (B, 32, D, H, W)

        # Ensure exact output size match (in case of odd dimensions)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(
                out, size=x.shape[2:], mode="trilinear", align_corners=False
            )

        logits = self.final_conv(out)  # (B, num_classes, D, H, W)

        return logits
