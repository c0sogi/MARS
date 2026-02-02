import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import CyclicConv2d, ResidualBlock2D


class SkyResUNet(nn.Module):
    """
    Cyclic Spatio-Temporal 2D Residual U-Net.

    This model processes 2D 'Sky Heatmaps' (Time x Azimuth) using cyclic convolutions
    along the Azimuth dimension to enforce angular continuity. It predicts 2D
    positional residuals (East, North) for each timestamp.

    Architecture:
    - Encoder: 4 stages of Residual Blocks + Max Pooling.
    - Bottleneck: High-capacity Residual Block.
    - Decoder: 4 stages of Upsampling + Concatenation + Residual Blocks.
    - Deep Supervision: Auxiliary heads at each decoder stage.
    """

    def __init__(self):
        super(SkyResUNet, self).__init__()

        self.in_channels = Config.INPUT_CHANNELS
        enc_ch = Config.ENCODER_CHANNELS  # [32, 64, 128, 256]

        # --- Encoder ---
        # Initial projection
        self.init_conv = CyclicConv2d(
            self.in_channels, enc_ch[0], kernel_size=3, padding=1
        )
        self.bn_init = nn.BatchNorm2d(enc_ch[0])
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.enc1 = ResidualBlock2D(enc_ch[0], enc_ch[0])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Stage 2
        self.enc2 = ResidualBlock2D(enc_ch[0], enc_ch[1])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Stage 3
        self.enc3 = ResidualBlock2D(enc_ch[1], enc_ch[2])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Stage 4
        self.enc4 = ResidualBlock2D(enc_ch[2], enc_ch[3])
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        # --- Bottleneck ---
        self.bottleneck = ResidualBlock2D(enc_ch[3], enc_ch[3] * 2)

        # --- Decoder ---
        # Decoder channels progression: 512 -> 256 -> 128 -> 64 -> 32

        # Stage 4 (Up from Bottleneck)
        # Input: Bottleneck (512) + Enc4 (256) = 768 -> Output: 256
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = ResidualBlock2D(enc_ch[3] * 2 + enc_ch[3], enc_ch[3])
        self.head4 = self._make_head(enc_ch[3])  # Deep Supervision

        # Stage 3
        # Input: Dec4 (256) + Enc3 (128) = 384 -> Output: 128
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ResidualBlock2D(enc_ch[3] + enc_ch[2], enc_ch[2])
        self.head3 = self._make_head(enc_ch[2])  # Deep Supervision

        # Stage 2
        # Input: Dec3 (128) + Enc2 (64) = 192 -> Output: 64
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ResidualBlock2D(enc_ch[2] + enc_ch[1], enc_ch[1])
        self.head2 = self._make_head(enc_ch[1])  # Deep Supervision

        # Stage 1
        # Input: Dec2 (64) + Enc1 (32) = 96 -> Output: 32
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ResidualBlock2D(enc_ch[1] + enc_ch[0], enc_ch[0])

        # --- Final Head ---
        self.final_head = self._make_head(enc_ch[0])

    def _make_head(self, in_features):
        """
        Creates a prediction head.
        Input: (B, C, T, A)
        Operation: Global Avg Pool over Azimuth -> (B, C, T) -> Transpose -> (B, T, C) -> Linear -> (B, T, 2)
        """
        return nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(inplace=True), nn.Linear(64, 2)
        )

    def _process_head(self, x, head_layer):
        """
        Applies the head layer to the feature map.
        x: (B, C, T, A)
        Returns: (B, T, 2)
        """
        # Global Average Pooling over Azimuth (dim 3)
        # x -> (B, C, T)
        x_pool = torch.mean(x, dim=3)

        # Permute to (B, T, C) for Linear layer
        x_perm = x_pool.permute(0, 2, 1)

        # Apply FC
        out = head_layer(x_perm)
        return out

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Time, Azimuth)

        Returns:
            If training: Dictionary of outputs {'main': ..., 'aux1': ..., ...}
            If eval: Tensor of shape (Batch, Time, 2) (Main output only)
        """
        # --- Encoder ---
        x = self.init_conv(x)
        x = self.bn_init(x)
        x = self.relu(x)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # --- Bottleneck ---
        b = self.bottleneck(p4)

        # --- Decoder ---
        # Stage 4
        d4 = self.up4(b)
        # Handle potential size mismatch due to odd dimensions during pooling
        if d4.size()[2:] != e4.size()[2:]:
            d4 = F.interpolate(
                d4, size=e4.size()[2:], mode="bilinear", align_corners=True
            )
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        # Stage 3
        d3 = self.up3(d4)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = F.interpolate(
                d3, size=e3.size()[2:], mode="bilinear", align_corners=True
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # Stage 2
        d2 = self.up2(d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = F.interpolate(
                d2, size=e2.size()[2:], mode="bilinear", align_corners=True
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # Stage 1
        d1 = self.up1(d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = F.interpolate(
                d1, size=e1.size()[2:], mode="bilinear", align_corners=True
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # --- Heads ---
        out_main = self._process_head(d1, self.final_head)

        if self.training:
            out_aux2 = self._process_head(d2, self.head2)
            out_aux3 = self._process_head(d3, self.head3)
            out_aux4 = self._process_head(d4, self.head4)

            return {
                "main": out_main,
                "aux2": out_aux2,
                "aux3": out_aux3,
                "aux4": out_aux4,
            }
        else:
            return out_main
