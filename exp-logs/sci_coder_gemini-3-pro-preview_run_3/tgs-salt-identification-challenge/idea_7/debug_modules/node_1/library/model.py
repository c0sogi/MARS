import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Module.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    U-Net++ Decoder Block.
    Performs Upsampling of the lower-level feature, Concatenation with skip connections,
    Convolution, and Attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # in_channels: Channels from the lower level (to be upsampled)
        # skip_channels: Sum of channels from all skip connections
        total_in_channels = in_channels + skip_channels

        self.conv1 = nn.Conv2d(
            total_in_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skips):
        """
        Args:
            x: Input tensor from the lower decoder level (or encoder bottleneck).
            skips: List of tensors from the same decoder level (skip connections).
        """
        # Upsample input from lower level
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate with skip connections
        if skips:
            # Ensure all tensors have the same spatial size before concat
            # (Handles potential rounding issues in odd dimensions, though 128 is power of 2)
            if x.shape[2:] != skips[0].shape[2:]:
                x = F.interpolate(
                    x, size=skips[0].shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat(skips + [x], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.scse(x)
        return x


class SaltUNetPlusPlus(nn.Module):
    """
    Nested U-Net (U-Net++) with ResNeXt-50 Encoder and Deep Supervision.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (ResNeXt-50 32x4d)
        # Features at strides: 2, 4, 8, 16, 32
        # Channels: [64, 256, 512, 1024, 2048]
        self.encoder = timm.create_model(
            Config.ENCODER,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Encoder channels
        enc_ch = [64, 256, 512, 1024, 2048]
        # Decoder channels (Design choice: 32, 64, 128, 256, 512)
        dec_ch = [32, 64, 128, 256, 512]

        # 2. Decoder Nodes
        # Naming convention: conv_{level}_{layer}
        # Level 0 (Top, Stride 2)
        self.conv_0_1 = DecoderBlock(enc_ch[1], enc_ch[0], dec_ch[0])
        self.conv_0_2 = DecoderBlock(dec_ch[1], enc_ch[0] + dec_ch[0], dec_ch[0])
        self.conv_0_3 = DecoderBlock(dec_ch[1], enc_ch[0] + 2 * dec_ch[0], dec_ch[0])
        self.conv_0_4 = DecoderBlock(dec_ch[1], enc_ch[0] + 3 * dec_ch[0], dec_ch[0])

        # Level 1 (Stride 4)
        self.conv_1_1 = DecoderBlock(enc_ch[2], enc_ch[1], dec_ch[1])
        self.conv_1_2 = DecoderBlock(dec_ch[2], enc_ch[1] + dec_ch[1], dec_ch[1])
        self.conv_1_3 = DecoderBlock(dec_ch[2], enc_ch[1] + 2 * dec_ch[1], dec_ch[1])

        # Level 2 (Stride 8)
        self.conv_2_1 = DecoderBlock(enc_ch[3], enc_ch[2], dec_ch[2])
        self.conv_2_2 = DecoderBlock(dec_ch[3], enc_ch[2] + dec_ch[2], dec_ch[2])

        # Level 3 (Stride 16)
        self.conv_3_1 = DecoderBlock(enc_ch[4], enc_ch[3], dec_ch[3])

        # 3. Segmentation Heads (Deep Supervision)
        # Attached to L0 nodes (Stride 2). Will be upsampled to Stride 1 in forward.
        self.final_convs = nn.ModuleList([nn.Conv2d(dec_ch[0], 1, 1) for _ in range(4)])

        self.deep_supervision = Config.DEEP_SUPERVISION

    def forward(self, x):
        # 1. Encoder Pass
        features = self.encoder(x)
        x_0_0 = features[0]  # Stride 2
        x_1_0 = features[1]  # Stride 4
        x_2_0 = features[2]  # Stride 8
        x_3_0 = features[3]  # Stride 16
        x_4_0 = features[4]  # Stride 32

        # 2. Decoder Pass (Nested Skip Pathways)

        # Column 1
        x_3_1 = self.conv_3_1(x_4_0, [x_3_0])
        x_2_1 = self.conv_2_1(x_3_0, [x_2_0])
        x_1_1 = self.conv_1_1(x_2_0, [x_1_0])
        x_0_1 = self.conv_0_1(x_1_0, [x_0_0])

        # Column 2
        x_2_2 = self.conv_2_2(x_3_1, [x_2_0, x_2_1])
        x_1_2 = self.conv_1_2(x_2_1, [x_1_0, x_1_1])
        x_0_2 = self.conv_0_2(x_1_1, [x_0_0, x_0_1])

        # Column 3
        x_1_3 = self.conv_1_3(x_2_2, [x_1_0, x_1_1, x_1_2])
        x_0_3 = self.conv_0_3(x_1_2, [x_0_0, x_0_1, x_0_2])

        # Column 4 (Final)
        x_0_4 = self.conv_0_4(x_1_3, [x_0_0, x_0_1, x_0_2, x_0_3])

        # 3. Output Heads
        # Nodes at L0 are Stride 2 (64x64). We need Stride 1 (128x128).
        outputs = []
        nodes = [x_0_1, x_0_2, x_0_3, x_0_4]

        for i, node in enumerate(nodes):
            # Apply 1x1 conv
            out = self.final_convs[i](node)
            # Upsample to original resolution
            out = F.interpolate(
                out, scale_factor=2, mode="bilinear", align_corners=True
            )
            outputs.append(out)

        if self.training and self.deep_supervision:
            # Return list of outputs for Deep Supervision loss
            return outputs
        else:
            # Return only the final output for validation/inference
            return outputs[-1]
