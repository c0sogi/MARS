import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UnetPlusPlus(nn.Module):
    """
    U-Net++ Architecture with EfficientNet-V2-M Backbone and Deep Supervision.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ):
        super().__init__()

        # 1. Encoder (EfficientNet-V2-M)
        # features_only=True returns features at strides [2, 4, 8, 16, 32]
        self.encoder = timm.create_model(
            backbone_name, pretrained=True, features_only=True, in_chans=in_channels
        )

        # Get encoder channel counts automatically
        # Typical EfficientNetV2-M channels: [24, 48, 80, 176, 512]
        enc_channels = self.encoder.feature_info.channels()

        # Decoder filter config (Lightweight to fit 768x768)
        # Levels corresponding to strides [2, 4, 8, 16, 32]
        self.filters = [32, 64, 128, 256, 512]

        # 2. Decoder Nodes
        # Notation: conv{row}_{col}
        # Row 0: Stride 2
        # Row 1: Stride 4
        # ...

        # --- Column 0 (Projection of Encoder Features) ---
        self.conv0_0 = ConvBlock(enc_channels[0], self.filters[0])
        self.conv1_0 = ConvBlock(enc_channels[1], self.filters[1])
        self.conv2_0 = ConvBlock(enc_channels[2], self.filters[2])
        self.conv3_0 = ConvBlock(enc_channels[3], self.filters[3])
        self.conv4_0 = ConvBlock(enc_channels[4], self.filters[4])

        # --- Column 1 ---
        # Inputs: Up(Row+1, Col-1) + (Row, Col-1)
        self.conv0_1 = ConvBlock(self.filters[0] + self.filters[1], self.filters[0])
        self.conv1_1 = ConvBlock(self.filters[1] + self.filters[2], self.filters[1])
        self.conv2_1 = ConvBlock(self.filters[2] + self.filters[3], self.filters[2])
        self.conv3_1 = ConvBlock(self.filters[3] + self.filters[4], self.filters[3])

        # --- Column 2 ---
        # Inputs: Up(Row+1, Col-1) + (Row, Col-1) + (Row, 0)
        # Dense skip connections accumulate channels from same row
        self.conv0_2 = ConvBlock(self.filters[0] * 2 + self.filters[1], self.filters[0])
        self.conv1_2 = ConvBlock(self.filters[1] * 2 + self.filters[2], self.filters[1])
        self.conv2_2 = ConvBlock(self.filters[2] * 2 + self.filters[3], self.filters[2])

        # --- Column 3 ---
        self.conv0_3 = ConvBlock(self.filters[0] * 3 + self.filters[1], self.filters[0])
        self.conv1_3 = ConvBlock(self.filters[1] * 3 + self.filters[2], self.filters[1])

        # --- Column 4 ---
        self.conv0_4 = ConvBlock(self.filters[0] * 4 + self.filters[1], self.filters[0])

        # 3. Segmentation Heads (Deep Supervision)
        # Attached to L0_1, L0_2, L0_3, L0_4
        self.final0_1 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        self.final0_2 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        self.final0_3 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        self.final0_4 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

        # Upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        # Input shape: (B, 3, H, W)
        img_h, img_w = x.shape[2], x.shape[3]

        # --- Encoder ---
        features = self.encoder(x)
        # x0: stride 2, x1: stride 4, x2: stride 8, x3: stride 16, x4: stride 32
        x0_0 = self.conv0_0(features[0])
        x1_0 = self.conv1_0(features[1])
        x2_0 = self.conv2_0(features[2])
        x3_0 = self.conv3_0(features[3])
        x4_0 = self.conv4_0(features[4])

        # --- Decoder Column 1 ---
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # --- Decoder Column 2 ---
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # --- Decoder Column 3 ---
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # --- Decoder Column 4 ---
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # --- Deep Supervision Outputs ---
        # All outputs are currently at stride 2. We upsample to stride 1 (Original Resolution).

        out0_4 = self.final0_4(x0_4)
        out0_3 = self.final0_3(x0_3)
        out0_2 = self.final0_2(x0_2)
        out0_1 = self.final0_1(x0_1)

        # Upsample to match input size
        # Note: We use interpolate directly to handle arbitrary input sizes exactly
        outputs = [out0_4, out0_3, out0_2, out0_1]
        upsampled_outputs = [
            F.interpolate(out, size=(img_h, img_w), mode="bilinear", align_corners=True)
            for out in outputs
        ]

        # Return list for deep supervision loss [Most Important, ..., Least Important]
        return upsampled_outputs
