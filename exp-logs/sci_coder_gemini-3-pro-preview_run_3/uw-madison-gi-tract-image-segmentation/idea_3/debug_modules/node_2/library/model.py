import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    (Conv3x3 -> BN -> ReLU) x 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegmentationModel(nn.Module):
    """
    U-Net++ (Nested U-Net) with EfficientNet-B4 Encoder.
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

        # Get channel counts from encoder feature maps
        # Typical EffNet-B4 channels: [24, 32, 56, 112, 160] (may vary by specific implementation/weights)
        # We use a dummy pass to get exact shapes if needed, but timm provides feature_info
        enc_channels = self.encoder.feature_info.channels()

        # Decoder filter sizes (decreasing as we go up)
        # We align these roughly with encoder channels or standard powers of 2
        # Level 0 (Stride 2), Level 1 (Stride 4), ...
        self.filters = [32, 64, 128, 256, 512]

        # 2. Decoder Blocks (Nested Skip Connections)

        # --- J=1 (First column of nested blocks) ---
        # Input: Encoder feature + Upsampled feature from below
        self.conv0_1 = ConvBlock(enc_channels[0] + enc_channels[1], self.filters[0])
        self.conv1_1 = ConvBlock(enc_channels[1] + enc_channels[2], self.filters[1])
        self.conv2_1 = ConvBlock(enc_channels[2] + enc_channels[3], self.filters[2])
        self.conv3_1 = ConvBlock(enc_channels[3] + enc_channels[4], self.filters[3])

        # --- J=2 (Second column) ---
        # Input: Encoder feature + Node_J1 + Upsampled feature from below
        # Note: Concatenation of [x0_0, x0_1, up(x1_1)]
        self.conv0_2 = ConvBlock(
            enc_channels[0] + self.filters[0] + self.filters[1], self.filters[0]
        )
        self.conv1_2 = ConvBlock(
            enc_channels[1] + self.filters[1] + self.filters[2], self.filters[1]
        )
        self.conv2_2 = ConvBlock(
            enc_channels[2] + self.filters[2] + self.filters[3], self.filters[2]
        )

        # --- J=3 (Third column) ---
        # Input: [x0_0, x0_1, x0_2, up(x1_2)]
        self.conv0_3 = ConvBlock(
            enc_channels[0] + 2 * self.filters[0] + self.filters[1], self.filters[0]
        )
        self.conv1_3 = ConvBlock(
            enc_channels[1] + 2 * self.filters[1] + self.filters[2], self.filters[1]
        )

        # --- J=4 (Fourth column - Final) ---
        # Input: [x0_0, x0_1, x0_2, x0_3, up(x1_3)]
        self.conv0_4 = ConvBlock(
            enc_channels[0] + 3 * self.filters[0] + self.filters[1], self.filters[0]
        )

        # 3. Final Segmentation Head
        self.final_conv = nn.Conv2d(self.filters[0], Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # Save input shape for final upsampling
        input_shape = x.shape[-2:]

        # --- Encoder ---
        # features is a list of tensors [x0_0, x1_0, x2_0, x3_0, x4_0]
        # Strides: 2, 4, 8, 16, 32
        features = self.encoder(x)
        x0_0 = features[0]
        x1_0 = features[1]
        x2_0 = features[2]
        x3_0 = features[3]
        x4_0 = features[4]

        # --- Decoder J=1 ---
        # Upsample lower layer and concat with current encoder layer
        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    F.interpolate(
                        x1_0, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    F.interpolate(
                        x2_0, size=x1_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    F.interpolate(
                        x3_0, size=x2_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    F.interpolate(
                        x4_0, size=x3_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Decoder J=2 ---
        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    F.interpolate(
                        x1_1, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    F.interpolate(
                        x2_1, size=x1_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    F.interpolate(
                        x3_1, size=x2_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Decoder J=3 ---
        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    F.interpolate(
                        x1_2, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    F.interpolate(
                        x2_2, size=x1_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Decoder J=4 ---
        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    F.interpolate(
                        x1_3, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Final Output ---
        # x0_4 is at stride 2. Upsample to stride 1 (input resolution).
        logits = self.final_conv(x0_4)
        logits = F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=True
        )

        return logits
