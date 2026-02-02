import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetPlusPlus(nn.Module):
    """
    U-Net++ Architecture with EfficientNet-B4 Encoder.

    Designed for 6-channel input (Ash Color + Temporal Diff).
    Implements the nested skip pathways to capture fine-grained spatial details.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder: EfficientNet-B4
        # in_chans=6 automatically adapts the first layer weights
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.INPUT_CHANNELS,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get encoder channel counts dynamically
        # For EfficientNet-B4, typically: [24, 32, 56, 160, 448]
        enc_ch = self.encoder.feature_info.channels()

        # Define Decoder Widths (number of channels at each level)
        # We use a tapered structure: 32, 48, 64, 128
        dec_ch = [32, 48, 64, 128]

        # --- Decoder Graph Construction ---

        # Level 3 (Stride 16)
        # Node X(3,1) <- [X(3,0), Up(X(4,0))]
        self.conv3_1 = ConvBlock(enc_ch[3] + enc_ch[4], dec_ch[3])

        # Level 2 (Stride 8)
        # Node X(2,1) <- [X(2,0), Up(X(3,0))]
        self.conv2_1 = ConvBlock(enc_ch[2] + enc_ch[3], dec_ch[2])
        # Node X(2,2) <- [X(2,0), X(2,1), Up(X(3,1))]
        self.conv2_2 = ConvBlock(enc_ch[2] + dec_ch[2] + dec_ch[3], dec_ch[2])

        # Level 1 (Stride 4)
        # Node X(1,1) <- [X(1,0), Up(X(2,0))]
        self.conv1_1 = ConvBlock(enc_ch[1] + enc_ch[2], dec_ch[1])
        # Node X(1,2) <- [X(1,0), X(1,1), Up(X(2,1))]
        self.conv1_2 = ConvBlock(enc_ch[1] + dec_ch[1] + dec_ch[2], dec_ch[1])
        # Node X(1,3) <- [X(1,0), X(1,1), X(1,2), Up(X(2,2))]
        self.conv1_3 = ConvBlock(enc_ch[1] + dec_ch[1] * 2 + dec_ch[2], dec_ch[1])

        # Level 0 (Stride 2)
        # Node X(0,1) <- [X(0,0), Up(X(1,0))]
        self.conv0_1 = ConvBlock(enc_ch[0] + enc_ch[1], dec_ch[0])
        # Node X(0,2) <- [X(0,0), X(0,1), Up(X(1,1))]
        self.conv0_2 = ConvBlock(enc_ch[0] + dec_ch[0] + dec_ch[1], dec_ch[0])
        # Node X(0,3) <- [X(0,0), X(0,1), X(0,2), Up(X(1,2))]
        self.conv0_3 = ConvBlock(enc_ch[0] + dec_ch[0] * 2 + dec_ch[1], dec_ch[0])
        # Node X(0,4) <- [X(0,0), X(0,1), X(0,2), X(0,3), Up(X(1,3))]
        self.conv0_4 = ConvBlock(enc_ch[0] + dec_ch[0] * 3 + dec_ch[1], dec_ch[0])

        # Final Segmentation Head
        self.final_conv = nn.Conv2d(dec_ch[0], 1, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Features: [e0, e1, e2, e3, e4]
        # Strides:  [2,  4,  8,  16, 32]
        enc_features = self.encoder(x)
        e0, e1, e2, e3, e4 = enc_features

        # Helper for bilinear upsampling
        def up(feat, target_shape):
            return F.interpolate(
                feat, size=target_shape, mode="bilinear", align_corners=False
            )

        # --- Decoder ---

        # Level 3
        # X(3,1)
        x3_1 = self.conv3_1(torch.cat([e3, up(e4, e3.shape[2:])], 1))

        # Level 2
        # X(2,1)
        x2_1 = self.conv2_1(torch.cat([e2, up(e3, e2.shape[2:])], 1))
        # X(2,2)
        x2_2 = self.conv2_2(torch.cat([e2, x2_1, up(x3_1, e2.shape[2:])], 1))

        # Level 1
        # X(1,1)
        x1_1 = self.conv1_1(torch.cat([e1, up(e2, e1.shape[2:])], 1))
        # X(1,2)
        x1_2 = self.conv1_2(torch.cat([e1, x1_1, up(x2_1, e1.shape[2:])], 1))
        # X(1,3)
        x1_3 = self.conv1_3(torch.cat([e1, x1_1, x1_2, up(x2_2, e1.shape[2:])], 1))

        # Level 0
        # X(0,1)
        x0_1 = self.conv0_1(torch.cat([e0, up(e1, e0.shape[2:])], 1))
        # X(0,2)
        x0_2 = self.conv0_2(torch.cat([e0, x0_1, up(x1_1, e0.shape[2:])], 1))
        # X(0,3)
        x0_3 = self.conv0_3(torch.cat([e0, x0_1, x0_2, up(x1_2, e0.shape[2:])], 1))
        # X(0,4) - Final Node
        x0_4 = self.conv0_4(
            torch.cat([e0, x0_1, x0_2, x0_3, up(x1_3, e0.shape[2:])], 1)
        )

        # --- Head ---
        logits = self.final_conv(x0_4)

        # Upsample from stride 2 (128x128) to original size (256x256)
        out = F.interpolate(
            logits,
            size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        return out
