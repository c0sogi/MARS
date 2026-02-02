import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard convolution block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UnetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) with EfficientNet-B4 backbone.
    Supports Deep Supervision.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (EfficientNet-B4)
        # features_only=True returns feature maps at strides [2, 4, 8, 16, 32]
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts from encoder
        # Typical EffNet-B4: [24, 32, 56, 160, 448]
        enc_channels = self.encoder.feature_info.channels()

        # 2. Decoder Configuration
        # Filter sizes for decoder levels 0 (stride 2) to 4 (stride 32)
        self.filters = [32, 64, 128, 256, 512]

        # 3. Nested Decoder Blocks (X_{i,j})
        # Naming convention: conv{row}_{col}

        # --- Column 1 (j=1) ---
        # Inputs: X_{i+1, 0} (upsampled) + X_{i, 0}
        self.conv0_1 = ConvBlock(enc_channels[1] + enc_channels[0], self.filters[0])
        self.conv1_1 = ConvBlock(enc_channels[2] + enc_channels[1], self.filters[1])
        self.conv2_1 = ConvBlock(enc_channels[3] + enc_channels[2], self.filters[2])
        self.conv3_1 = ConvBlock(enc_channels[4] + enc_channels[3], self.filters[3])

        # --- Column 2 (j=2) ---
        # Inputs: X_{i+1, 1} (upsampled) + X_{i, 0} + X_{i, 1}
        self.conv0_2 = ConvBlock(
            self.filters[1] + enc_channels[0] + self.filters[0], self.filters[0]
        )
        self.conv1_2 = ConvBlock(
            self.filters[2] + enc_channels[1] + self.filters[1], self.filters[1]
        )
        self.conv2_2 = ConvBlock(
            self.filters[3] + enc_channels[2] + self.filters[2], self.filters[2]
        )

        # --- Column 3 (j=3) ---
        # Inputs: X_{i+1, 2} (upsampled) + X_{i, 0} + X_{i, 1} + X_{i, 2}
        self.conv0_3 = ConvBlock(
            self.filters[1] + enc_channels[0] + 2 * self.filters[0], self.filters[0]
        )
        self.conv1_3 = ConvBlock(
            self.filters[2] + enc_channels[1] + 2 * self.filters[1], self.filters[1]
        )

        # --- Column 4 (j=4) ---
        # Inputs: X_{i+1, 3} (upsampled) + X_{i, 0} + X_{i, 1} + X_{i, 2} + X_{i, 3}
        self.conv0_4 = ConvBlock(
            self.filters[1] + enc_channels[0] + 3 * self.filters[0], self.filters[0]
        )

        # 4. Deep Supervision Heads
        # All output heads originate from Row 0 (Stride 2)
        self.final1 = nn.Conv2d(self.filters[0], Config.NUM_CLASSES, kernel_size=1)
        self.final2 = nn.Conv2d(self.filters[0], Config.NUM_CLASSES, kernel_size=1)
        self.final3 = nn.Conv2d(self.filters[0], Config.NUM_CLASSES, kernel_size=1)
        self.final4 = nn.Conv2d(self.filters[0], Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, C, H, W)
        img_h, img_w = x.shape[2], x.shape[3]

        # --- Encoder ---
        features = self.encoder(x)
        x0_0 = features[0]  # Stride 2
        x1_0 = features[1]  # Stride 4
        x2_0 = features[2]  # Stride 8
        x3_0 = features[3]  # Stride 16
        x4_0 = features[4]  # Stride 32

        # --- Decoder Column 1 ---
        x1_0_up = F.interpolate(
            x1_0, size=x0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_0_up], dim=1))

        x2_0_up = F.interpolate(
            x2_0, size=x1_0.shape[2:], mode="bilinear", align_corners=True
        )
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_0_up], dim=1))

        x3_0_up = F.interpolate(
            x3_0, size=x2_0.shape[2:], mode="bilinear", align_corners=True
        )
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_0_up], dim=1))

        x4_0_up = F.interpolate(
            x4_0, size=x3_0.shape[2:], mode="bilinear", align_corners=True
        )
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0_up], dim=1))

        # --- Decoder Column 2 ---
        x1_1_up = F.interpolate(
            x1_1, size=x0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1_up], dim=1))

        x2_1_up = F.interpolate(
            x2_1, size=x1_0.shape[2:], mode="bilinear", align_corners=True
        )
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1_up], dim=1))

        x3_1_up = F.interpolate(
            x3_1, size=x2_0.shape[2:], mode="bilinear", align_corners=True
        )
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1_up], dim=1))

        # --- Decoder Column 3 ---
        x1_2_up = F.interpolate(
            x1_2, size=x0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_up], dim=1))

        x2_2_up = F.interpolate(
            x2_2, size=x1_0.shape[2:], mode="bilinear", align_corners=True
        )
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2_up], dim=1))

        # --- Decoder Column 4 ---
        x1_3_up = F.interpolate(
            x1_3, size=x0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_up], dim=1))

        # --- Output Heads ---
        # The outputs are at Stride 2 (size of x0_0). We upsample to Stride 1 (Input Size).

        out4 = self.final4(x0_4)
        out4 = F.interpolate(
            out4, size=(img_h, img_w), mode="bilinear", align_corners=True
        )

        if self.training:
            # Deep Supervision: Return list of outputs
            out1 = self.final1(x0_1)
            out1 = F.interpolate(
                out1, size=(img_h, img_w), mode="bilinear", align_corners=True
            )

            out2 = self.final2(x0_2)
            out2 = F.interpolate(
                out2, size=(img_h, img_w), mode="bilinear", align_corners=True
            )

            out3 = self.final3(x0_3)
            out3 = F.interpolate(
                out3, size=(img_h, img_w), mode="bilinear", align_corners=True
            )

            return [out1, out2, out3, out4]

        # Inference: Return only the final output
        return out4
