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
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UnetPlusPlus(nn.Module):
    """
    U-Net++ Architecture with ResNet-34 Backbone.
    Features nested skip pathways and deep supervision.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        classes=Config.NUM_CLASSES,
        deep_supervision=Config.DEEP_SUPERVISION,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.classes = classes

        # Load Backbone (ResNet-34)
        # features_only=True returns a list of feature maps at different strides
        # ResNet34 features:
        # 0: Stride 2, 64 ch
        # 1: Stride 4, 64 ch
        # 2: Stride 8, 128 ch
        # 3: Stride 16, 256 ch
        # 4: Stride 32, 512 ch
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Encoder Channels for ResNet34
        enc_channels = [64, 64, 128, 256, 512]

        # --- Decoder Blocks ---
        # We maintain the same channel width as the encoder at each level for the decoder nodes.

        # Level 0 (Output Stride 2)
        # x_0_0 is encoder[0]
        # x_0_1: cat(x_0_0, up(x_1_0)) -> 64 + 64 = 128 in
        self.conv_0_1 = ConvBlock(enc_channels[0] + enc_channels[1], enc_channels[0])
        # x_0_2: cat(x_0_0, x_0_1, up(x_1_1)) -> 64 + 64 + 64 = 192 in
        self.conv_0_2 = ConvBlock(
            enc_channels[0] * 2 + enc_channels[1], enc_channels[0]
        )
        # x_0_3: cat(x_0_0, x_0_1, x_0_2, up(x_1_2)) -> 64*3 + 64 = 256 in
        self.conv_0_3 = ConvBlock(
            enc_channels[0] * 3 + enc_channels[1], enc_channels[0]
        )
        # x_0_4: cat(x_0_0, x_0_1, x_0_2, x_0_3, up(x_1_3)) -> 64*4 + 64 = 320 in
        self.conv_0_4 = ConvBlock(
            enc_channels[0] * 4 + enc_channels[1], enc_channels[0]
        )

        # Level 1 (Output Stride 4)
        # x_1_0 is encoder[1]
        # x_1_1: cat(x_1_0, up(x_2_0)) -> 64 + 128 = 192 in
        self.conv_1_1 = ConvBlock(enc_channels[1] + enc_channels[2], enc_channels[1])
        # x_1_2: cat(x_1_0, x_1_1, up(x_2_1)) -> 64 + 64 + 128 = 256 in
        self.conv_1_2 = ConvBlock(
            enc_channels[1] * 2 + enc_channels[2], enc_channels[1]
        )
        # x_1_3: cat(x_1_0, x_1_1, x_1_2, up(x_2_2)) -> 64*3 + 128 = 320 in
        self.conv_1_3 = ConvBlock(
            enc_channels[1] * 3 + enc_channels[2], enc_channels[1]
        )

        # Level 2 (Output Stride 8)
        # x_2_0 is encoder[2]
        # x_2_1: cat(x_2_0, up(x_3_0)) -> 128 + 256 = 384 in
        self.conv_2_1 = ConvBlock(enc_channels[2] + enc_channels[3], enc_channels[2])
        # x_2_2: cat(x_2_0, x_2_1, up(x_3_1)) -> 128 + 128 + 256 = 512 in
        self.conv_2_2 = ConvBlock(
            enc_channels[2] * 2 + enc_channels[3], enc_channels[2]
        )

        # Level 3 (Output Stride 16)
        # x_3_0 is encoder[3]
        # x_3_1: cat(x_3_0, up(x_4_0)) -> 256 + 512 = 768 in
        self.conv_3_1 = ConvBlock(enc_channels[3] + enc_channels[4], enc_channels[3])

        # Level 4 is just encoder[4] (Stride 32)

        # --- Final Output Heads ---
        # We output from the Level 0 nodes (Stride 2).
        if self.deep_supervision:
            self.final_1 = nn.Conv2d(enc_channels[0], classes, kernel_size=1)
            self.final_2 = nn.Conv2d(enc_channels[0], classes, kernel_size=1)
            self.final_3 = nn.Conv2d(enc_channels[0], classes, kernel_size=1)
            self.final_4 = nn.Conv2d(enc_channels[0], classes, kernel_size=1)
        else:
            self.final_4 = nn.Conv2d(enc_channels[0], classes, kernel_size=1)

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]

        # 1. Encoder Forward
        features = self.backbone(x)
        x_0_0 = features[0]  # Stride 2
        x_1_0 = features[1]  # Stride 4
        x_2_0 = features[2]  # Stride 8
        x_3_0 = features[3]  # Stride 16
        x_4_0 = features[4]  # Stride 32

        # 2. Decoder Forward (Nested Pathways)

        # --- Column j=1 ---
        # x_0_1
        x_1_0_up = F.interpolate(
            x_1_0, size=x_0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_0_1 = self.conv_0_1(torch.cat([x_0_0, x_1_0_up], dim=1))

        # x_1_1
        x_2_0_up = F.interpolate(
            x_2_0, size=x_1_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, x_2_0_up], dim=1))

        # x_2_1
        x_3_0_up = F.interpolate(
            x_3_0, size=x_2_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, x_3_0_up], dim=1))

        # x_3_1
        x_4_0_up = F.interpolate(
            x_4_0, size=x_3_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_3_1 = self.conv_3_1(torch.cat([x_3_0, x_4_0_up], dim=1))

        # --- Column j=2 ---
        # x_0_2
        x_1_1_up = F.interpolate(
            x_1_1, size=x_0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, x_1_1_up], dim=1))

        # x_1_2
        x_2_1_up = F.interpolate(
            x_2_1, size=x_1_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, x_2_1_up], dim=1))

        # x_2_2
        x_3_1_up = F.interpolate(
            x_3_1, size=x_2_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_2_2 = self.conv_2_2(torch.cat([x_2_0, x_2_1, x_3_1_up], dim=1))

        # --- Column j=3 ---
        # x_0_3
        x_1_2_up = F.interpolate(
            x_1_2, size=x_0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_0_3 = self.conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, x_1_2_up], dim=1))

        # x_1_3
        x_2_2_up = F.interpolate(
            x_2_2, size=x_1_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_1_3 = self.conv_1_3(torch.cat([x_1_0, x_1_1, x_1_2, x_2_2_up], dim=1))

        # --- Column j=4 ---
        # x_0_4
        x_1_3_up = F.interpolate(
            x_1_3, size=x_0_0.shape[2:], mode="bilinear", align_corners=True
        )
        x_0_4 = self.conv_0_4(torch.cat([x_0_0, x_0_1, x_0_2, x_0_3, x_1_3_up], dim=1))

        # 3. Output Heads
        # All outputs are currently at Stride 2. We interpolate to Stride 1 (Input Size).

        if self.deep_supervision and self.training:
            out_1 = self.final_1(x_0_1)
            out_2 = self.final_2(x_0_2)
            out_3 = self.final_3(x_0_3)
            out_4 = self.final_4(x_0_4)

            out_1 = F.interpolate(
                out_1, size=(img_h, img_w), mode="bilinear", align_corners=True
            )
            out_2 = F.interpolate(
                out_2, size=(img_h, img_w), mode="bilinear", align_corners=True
            )
            out_3 = F.interpolate(
                out_3, size=(img_h, img_w), mode="bilinear", align_corners=True
            )
            out_4 = F.interpolate(
                out_4, size=(img_h, img_w), mode="bilinear", align_corners=True
            )

            return [out_1, out_2, out_3, out_4]
        else:
            # During inference or validation, use the final, most refined node
            out = self.final_4(x_0_4)
            out = F.interpolate(
                out, size=(img_h, img_w), mode="bilinear", align_corners=True
            )
            return out
