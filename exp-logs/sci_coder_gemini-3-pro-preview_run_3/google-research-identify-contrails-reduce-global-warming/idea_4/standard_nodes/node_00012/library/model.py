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


class ContrailUNetPlusPlus(nn.Module):
    """
    U-Net++ with EfficientNet Encoder (Config.BACKBONE) and Deep Supervision.

    Architecture:
    - Encoder: EfficientNet (pretrained, features_only)
    - Decoder: Nested U-Net (U-Net++) with dense skip connections
    - Input: 6 Channels (Ash Color + Temporal Diff)
    - Output: 1 Channel (Binary Logits)
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        # Load EfficientNet-B4 with 6 input channels
        # features_only=True returns feature maps at strides 2, 4, 8, 16, 32
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Get encoder channel counts
        enc_channels = self.encoder.feature_info.channels()

        # Define Decoder Channels (Hyperparameters)
        # We decrease channels as we go up the decoder path
        # Corresponds to rows 0, 1, 2, 3 (Strides 2, 4, 8, 16)
        self.dec_channels = [32, 64, 128, 256]

        # --- Nested Decoder Blocks ---
        # Notation: conv_{row}_{col}
        # Row 0: Stride 2
        # Row 1: Stride 4
        # Row 2: Stride 8
        # Row 3: Stride 16

        # Column 1 (Standard U-Net decoder blocks)
        # Inputs: Up(x{row+1}_0) and x{row}_0
        self.conv_0_1 = ConvBlock(
            enc_channels[0] + enc_channels[1], self.dec_channels[0]
        )
        self.conv_1_1 = ConvBlock(
            enc_channels[1] + enc_channels[2], self.dec_channels[1]
        )
        self.conv_2_1 = ConvBlock(
            enc_channels[2] + enc_channels[3], self.dec_channels[2]
        )
        self.conv_3_1 = ConvBlock(
            enc_channels[3] + enc_channels[4], self.dec_channels[3]
        )

        # Column 2
        # Inputs: Up(x{row+1}_1), x{row}_0, x{row}_1
        self.conv_0_2 = ConvBlock(
            enc_channels[0] + self.dec_channels[0] + self.dec_channels[1],
            self.dec_channels[0],
        )
        self.conv_1_2 = ConvBlock(
            enc_channels[1] + self.dec_channels[1] + self.dec_channels[2],
            self.dec_channels[1],
        )
        self.conv_2_2 = ConvBlock(
            enc_channels[2] + self.dec_channels[2] + self.dec_channels[3],
            self.dec_channels[2],
        )

        # Column 3
        # Inputs: Up(x{row+1}_2), x{row}_0, x{row}_1, x{row}_2
        self.conv_0_3 = ConvBlock(
            enc_channels[0] + 2 * self.dec_channels[0] + self.dec_channels[1],
            self.dec_channels[0],
        )
        self.conv_1_3 = ConvBlock(
            enc_channels[1] + 2 * self.dec_channels[1] + self.dec_channels[2],
            self.dec_channels[1],
        )

        # Column 4 (Final Output Column)
        # Inputs: Up(x1_3), x0_0, x0_1, x0_2, x0_3
        self.conv_0_4 = ConvBlock(
            enc_channels[0] + 3 * self.dec_channels[0] + self.dec_channels[1],
            self.dec_channels[0],
        )

        # --- Deep Supervision Heads ---
        # 1x1 Convs to project to 1 channel (logit)
        # Attached to x0_1, x0_2, x0_3, x0_4
        self.final_0_1 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)
        self.final_0_2 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)
        self.final_0_3 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)
        self.final_0_4 = nn.Conv2d(self.dec_channels[0], 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        x0_0, x1_0, x2_0, x3_0, x4_0 = features
        # x0_0: Stride 2
        # x1_0: Stride 4
        # x2_0: Stride 8
        # x3_0: Stride 16
        # x4_0: Stride 32

        # Decoder Column 1
        x0_1 = self.conv_0_1(
            torch.cat(
                [
                    F.interpolate(
                        x1_0, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x0_0,
                ],
                dim=1,
            )
        )
        x1_1 = self.conv_1_1(
            torch.cat(
                [
                    F.interpolate(
                        x2_0, size=x1_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x1_0,
                ],
                dim=1,
            )
        )
        x2_1 = self.conv_2_1(
            torch.cat(
                [
                    F.interpolate(
                        x3_0, size=x2_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x2_0,
                ],
                dim=1,
            )
        )
        x3_1 = self.conv_3_1(
            torch.cat(
                [
                    F.interpolate(
                        x4_0, size=x3_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x3_0,
                ],
                dim=1,
            )
        )

        # Decoder Column 2
        x0_2 = self.conv_0_2(
            torch.cat(
                [
                    F.interpolate(
                        x1_1, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x0_0,
                    x0_1,
                ],
                dim=1,
            )
        )
        x1_2 = self.conv_1_2(
            torch.cat(
                [
                    F.interpolate(
                        x2_1, size=x1_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x1_0,
                    x1_1,
                ],
                dim=1,
            )
        )
        x2_2 = self.conv_2_2(
            torch.cat(
                [
                    F.interpolate(
                        x3_1, size=x2_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x2_0,
                    x2_1,
                ],
                dim=1,
            )
        )

        # Decoder Column 3
        x0_3 = self.conv_0_3(
            torch.cat(
                [
                    F.interpolate(
                        x1_2, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x0_0,
                    x0_1,
                    x0_2,
                ],
                dim=1,
            )
        )
        x1_3 = self.conv_1_3(
            torch.cat(
                [
                    F.interpolate(
                        x2_2, size=x1_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x1_0,
                    x1_1,
                    x1_2,
                ],
                dim=1,
            )
        )

        # Decoder Column 4
        x0_4 = self.conv_0_4(
            torch.cat(
                [
                    F.interpolate(
                        x1_3, size=x0_0.shape[2:], mode="bilinear", align_corners=True
                    ),
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                ],
                dim=1,
            )
        )

        # Heads & Final Upsampling (Stride 2 -> Stride 1)
        # We upsample the logits to match input resolution (256x256)
        logit_0_4 = F.interpolate(
            self.final_0_4(x0_4), scale_factor=2, mode="bilinear", align_corners=True
        )

        if self.training:
            # Deep Supervision: Return logits from all intermediate nodes at level 0
            logit_0_1 = F.interpolate(
                self.final_0_1(x0_1),
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            logit_0_2 = F.interpolate(
                self.final_0_2(x0_2),
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            logit_0_3 = F.interpolate(
                self.final_0_3(x0_3),
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            # Return list for Deep Supervision loss computation (Deepest to Shallowest priority)
            return [logit_0_4, logit_0_3, logit_0_2, logit_0_1]
        else:
            # Inference: Return only the final output
            return logit_0_4
