import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()

        # Channel Squeeze and Excitation (cSE)
        # Squeeze: Global Average Pooling
        # Excitation: Dense -> ReLU -> Dense -> Sigmoid
        mid_channels = max(1, in_channels // reduction)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial Squeeze and Excitation (sSE)
        # Squeeze: 1x1 Conv -> Sigmoid
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent: Add channel-wise and spatial-wise recalibrations
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Convolution Block for U-Net++ Decoder Nodes.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> scSE
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.scse(x)
        return x


class SaltSegModel(nn.Module):
    """
    U-Net++ with ResNeXt-50 (32x4d) Encoder and scSE Attention.
    Implements Deep Supervision and Input Channel Multiplexing.
    """

    def __init__(self, encoder_name=Config.ENCODER_NAME, pretrained=True):
        super(SaltSegModel, self).__init__()

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        # Load pretrained ResNeXt-50
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained, features_only=True, in_chans=3
        )

        # Get encoder channel counts
        # For resnext50_32x4d, typically: [64, 256, 512, 1024, 2048]
        # Indices: 0 (Stride 2), 1 (Stride 4), 2 (Stride 8), 3 (Stride 16), 4 (Stride 32)
        enc_channels = self.encoder.feature_info.channels()

        # ---------------------------------------------------------------------
        # Decoder Configuration
        # ---------------------------------------------------------------------
        # Target channels for decoder levels L0, L1, L2, L3
        # L0 corresponds to Stride 2 (Output level)
        self.dec_channels = [32, 64, 128, 256]

        # ---------------------------------------------------------------------
        # Decoder Nodes (U-Net++ Dense Connections)
        # ---------------------------------------------------------------------

        # Layer 3 Nodes (Stride 16)
        # Inputs: Encoder L3, Upsampled L4
        self.conv_3_1 = ConvBlock(
            enc_channels[3] + enc_channels[4], self.dec_channels[3]
        )

        # Layer 2 Nodes (Stride 8)
        # Node 2_1: Inputs Encoder L2, Upsampled L3
        self.conv_2_1 = ConvBlock(
            enc_channels[2] + enc_channels[3], self.dec_channels[2]
        )
        # Node 2_2: Inputs Encoder L2, Node 2_1, Upsampled Node 3_1
        self.conv_2_2 = ConvBlock(
            enc_channels[2] + self.dec_channels[2] + self.dec_channels[3],
            self.dec_channels[2],
        )

        # Layer 1 Nodes (Stride 4)
        # Node 1_1: Inputs Encoder L1, Upsampled L2
        self.conv_1_1 = ConvBlock(
            enc_channels[1] + enc_channels[2], self.dec_channels[1]
        )
        # Node 1_2: Inputs Encoder L1, Node 1_1, Upsampled Node 2_1
        self.conv_1_2 = ConvBlock(
            enc_channels[1] + self.dec_channels[1] + self.dec_channels[2],
            self.dec_channels[1],
        )
        # Node 1_3: Inputs Encoder L1, Node 1_1, Node 1_2, Upsampled Node 2_2
        self.conv_1_3 = ConvBlock(
            enc_channels[1] + self.dec_channels[1] * 2 + self.dec_channels[2],
            self.dec_channels[1],
        )

        # Layer 0 Nodes (Stride 2) - Top Level
        # Node 0_1: Inputs Encoder L0, Upsampled L1
        self.conv_0_1 = ConvBlock(
            enc_channels[0] + enc_channels[1], self.dec_channels[0]
        )
        # Node 0_2: Inputs Encoder L0, Node 0_1, Upsampled Node 1_1
        self.conv_0_2 = ConvBlock(
            enc_channels[0] + self.dec_channels[0] + self.dec_channels[1],
            self.dec_channels[0],
        )
        # Node 0_3: Inputs Encoder L0, Node 0_1, Node 0_2, Upsampled Node 1_2
        self.conv_0_3 = ConvBlock(
            enc_channels[0] + self.dec_channels[0] * 2 + self.dec_channels[1],
            self.dec_channels[0],
        )
        # Node 0_4: Inputs Encoder L0, Node 0_1, Node 0_2, Node 0_3, Upsampled Node 1_3
        self.conv_0_4 = ConvBlock(
            enc_channels[0] + self.dec_channels[0] * 3 + self.dec_channels[1],
            self.dec_channels[0],
        )

        # ---------------------------------------------------------------------
        # Deep Supervision Heads
        # ---------------------------------------------------------------------
        # 1x1 Convs to map feature channels to 1 output class
        self.final_0_1 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)
        self.final_0_2 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)
        self.final_0_3 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)
        self.final_0_4 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)

    def forward(self, x):
        # ---------------------------------------------------------------------
        # Input Multiplexing
        # ---------------------------------------------------------------------
        # If input is (B, 2, H, W) -> [Seismic, Depth], convert to (B, 3, H, W)
        # Strategy: [Seismic, Seismic, Depth]
        if x.size(1) == 2:
            x = torch.cat([x[:, 0:1, :, :], x[:, 0:1, :, :], x[:, 1:2, :, :]], dim=1)

        input_size = x.shape[-2:]  # (H, W)

        # ---------------------------------------------------------------------
        # Encoder Pass
        # ---------------------------------------------------------------------
        features = self.encoder(x)
        x_0_0 = features[0]  # Stride 2
        x_1_0 = features[1]  # Stride 4
        x_2_0 = features[2]  # Stride 8
        x_3_0 = features[3]  # Stride 16
        x_4_0 = features[4]  # Stride 32

        # ---------------------------------------------------------------------
        # Decoder Pass (Nested)
        # ---------------------------------------------------------------------

        # Layer 3
        # Upsample x_4_0 to match x_3_0
        up_x_4_0 = F.interpolate(
            x_4_0, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_3_1 = self.conv_3_1(torch.cat([x_3_0, up_x_4_0], dim=1))

        # Layer 2
        up_x_3_0 = F.interpolate(
            x_3_0, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, up_x_3_0], dim=1))

        up_x_3_1 = F.interpolate(
            x_3_1, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_2_2 = self.conv_2_2(torch.cat([x_2_0, x_2_1, up_x_3_1], dim=1))

        # Layer 1
        up_x_2_0 = F.interpolate(
            x_2_0, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, up_x_2_0], dim=1))

        up_x_2_1 = F.interpolate(
            x_2_1, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, up_x_2_1], dim=1))

        up_x_2_2 = F.interpolate(
            x_2_2, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_1_3 = self.conv_1_3(torch.cat([x_1_0, x_1_1, x_1_2, up_x_2_2], dim=1))

        # Layer 0 (Top Level)
        up_x_1_0 = F.interpolate(
            x_1_0, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_0_1 = self.conv_0_1(torch.cat([x_0_0, up_x_1_0], dim=1))

        up_x_1_1 = F.interpolate(
            x_1_1, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, up_x_1_1], dim=1))

        up_x_1_2 = F.interpolate(
            x_1_2, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_0_3 = self.conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, up_x_1_2], dim=1))

        up_x_1_3 = F.interpolate(
            x_1_3, scale_factor=2, mode="bilinear", align_corners=True
        )
        x_0_4 = self.conv_0_4(torch.cat([x_0_0, x_0_1, x_0_2, x_0_3, up_x_1_3], dim=1))

        # ---------------------------------------------------------------------
        # Deep Supervision Output
        # ---------------------------------------------------------------------
        # Apply 1x1 convs
        logits_0_1 = self.final_0_1(x_0_1)
        logits_0_2 = self.final_0_2(x_0_2)
        logits_0_3 = self.final_0_3(x_0_3)
        logits_0_4 = self.final_0_4(x_0_4)

        # Upsample all outputs to original input size
        # x_0_j are at Stride 2 (e.g., 64x64 for 128x128 input)
        outputs = []
        for l in [logits_0_1, logits_0_2, logits_0_3, logits_0_4]:
            outputs.append(
                F.interpolate(l, size=input_size, mode="bilinear", align_corners=True)
            )

        return outputs
