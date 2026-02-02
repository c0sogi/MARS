import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    Standard Decoder Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> scSE
    """

    def __init__(self, in_channels, out_channels, use_scse=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_scse = use_scse
        if use_scse:
            self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_scse:
            x = self.scse(x)
        return x


class SaltUNetPlusPlus(nn.Module):
    def __init__(
        self,
        encoder_name=Config.ENCODER_NAME,
        in_channels=Config.IN_CHANNELS,
        deep_supervision=Config.DEEP_SUPERVISION,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Load Encoder
        # features_only=True returns a list of feature maps
        # out_indices=(0, 1, 2, 3, 4) corresponds to stride 2, 4, 8, 16, 32
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # 2. Modify first layer for 4 channels (RGB + Depth)
        if in_channels != 3:
            old_conv = self.encoder.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            # Initialize weights
            w = old_conv.weight.data
            new_w = new_conv.weight.data
            # Copy RGB weights
            new_w[:, :3, :, :] = w
            # Initialize extra channels with mean of RGB weights
            for i in range(3, in_channels):
                new_w[:, i, :, :] = w.mean(dim=1)

            self.encoder.conv1 = new_conv

        # 3. Define Channels
        # Encoder Channels (ResNeXt-50): [64, 256, 512, 1024, 2048]
        enc_ch = self.encoder.feature_info.channels()

        # Decoder Channels (Design choice for capacity)
        dec_ch = [64, 128, 256, 512]

        # 4. Construct U-Net++ Graph
        # Node naming: conv_{row}_{col}
        # Rows: 0 to 3 (corresponding to enc_ch[0] to enc_ch[3])
        # Cols: 1 to 4 (nesting level)

        # --- Column 1 (Standard U-Net connections) ---
        # X_3_1: Input [e3, Up(e4)]
        self.conv_3_1 = DecoderBlock(enc_ch[3] + enc_ch[4], dec_ch[3])

        # X_2_1: Input [e2, Up(e3)]
        self.conv_2_1 = DecoderBlock(enc_ch[2] + enc_ch[3], dec_ch[2])

        # X_1_1: Input [e1, Up(e2)]
        self.conv_1_1 = DecoderBlock(enc_ch[1] + enc_ch[2], dec_ch[1])

        # X_0_1: Input [e0, Up(e1)]
        self.conv_0_1 = DecoderBlock(enc_ch[0] + enc_ch[1], dec_ch[0])

        # --- Column 2 (Nested connections) ---
        # X_2_2: Input [e2, X_2_1, Up(X_3_1)]
        self.conv_2_2 = DecoderBlock(enc_ch[2] + dec_ch[2] + dec_ch[3], dec_ch[2])

        # X_1_2: Input [e1, X_1_1, Up(X_2_1)]
        self.conv_1_2 = DecoderBlock(enc_ch[1] + dec_ch[1] + dec_ch[2], dec_ch[1])

        # X_0_2: Input [e0, X_0_1, Up(X_1_1)]
        self.conv_0_2 = DecoderBlock(enc_ch[0] + dec_ch[0] + dec_ch[1], dec_ch[0])

        # --- Column 3 (Nested connections) ---
        # X_1_3: Input [e1, X_1_1, X_1_2, Up(X_2_2)]
        self.conv_1_3 = DecoderBlock(enc_ch[1] + dec_ch[1] * 2 + dec_ch[2], dec_ch[1])

        # X_0_3: Input [e0, X_0_1, X_0_2, Up(X_1_2)]
        self.conv_0_3 = DecoderBlock(enc_ch[0] + dec_ch[0] * 2 + dec_ch[1], dec_ch[0])

        # --- Column 4 (Final Nested connections) ---
        # X_0_4: Input [e0, X_0_1, X_0_2, X_0_3, Up(X_1_3)]
        self.conv_0_4 = DecoderBlock(enc_ch[0] + dec_ch[0] * 3 + dec_ch[1], dec_ch[0])

        # 5. Final Segmentation Heads (1x1 Convs)
        self.final_0_1 = nn.Conv2d(dec_ch[0], 1, kernel_size=1)
        self.final_0_2 = nn.Conv2d(dec_ch[0], 1, kernel_size=1)
        self.final_0_3 = nn.Conv2d(dec_ch[0], 1, kernel_size=1)
        self.final_0_4 = nn.Conv2d(dec_ch[0], 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        e0, e1, e2, e3, e4 = features
        # e0: Stride 2
        # e1: Stride 4
        # e2: Stride 8
        # e3: Stride 16
        # e4: Stride 32

        # Decoder Nested Steps

        # --- Column 1 ---
        x_3_1 = self.conv_3_1(
            torch.cat(
                [
                    e3,
                    F.interpolate(
                        e4, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x_2_1 = self.conv_2_1(
            torch.cat(
                [
                    e2,
                    F.interpolate(
                        e3, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x_1_1 = self.conv_1_1(
            torch.cat(
                [
                    e1,
                    F.interpolate(
                        e2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x_0_1 = self.conv_0_1(
            torch.cat(
                [
                    e0,
                    F.interpolate(
                        e1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Column 2 ---
        x_2_2 = self.conv_2_2(
            torch.cat(
                [
                    e2,
                    x_2_1,
                    F.interpolate(
                        x_3_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x_1_2 = self.conv_1_2(
            torch.cat(
                [
                    e1,
                    x_1_1,
                    F.interpolate(
                        x_2_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x_0_2 = self.conv_0_2(
            torch.cat(
                [
                    e0,
                    x_0_1,
                    F.interpolate(
                        x_1_1, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Column 3 ---
        x_1_3 = self.conv_1_3(
            torch.cat(
                [
                    e1,
                    x_1_1,
                    x_1_2,
                    F.interpolate(
                        x_2_2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        x_0_3 = self.conv_0_3(
            torch.cat(
                [
                    e0,
                    x_0_1,
                    x_0_2,
                    F.interpolate(
                        x_1_2, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # --- Column 4 ---
        x_0_4 = self.conv_0_4(
            torch.cat(
                [
                    e0,
                    x_0_1,
                    x_0_2,
                    x_0_3,
                    F.interpolate(
                        x_1_3, scale_factor=2, mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )

        # Outputs
        # The decoder outputs (x_0_j) are at Stride 2 (relative to input).
        # We must upsample by 2 to get back to original resolution.

        out_0_1 = F.interpolate(
            self.final_0_1(x_0_1), scale_factor=2, mode="bilinear", align_corners=True
        )
        out_0_2 = F.interpolate(
            self.final_0_2(x_0_2), scale_factor=2, mode="bilinear", align_corners=True
        )
        out_0_3 = F.interpolate(
            self.final_0_3(x_0_3), scale_factor=2, mode="bilinear", align_corners=True
        )
        out_0_4 = F.interpolate(
            self.final_0_4(x_0_4), scale_factor=2, mode="bilinear", align_corners=True
        )

        if self.deep_supervision and self.training:
            return [out_0_1, out_0_2, out_0_3, out_0_4]
        else:
            return out_0_4
