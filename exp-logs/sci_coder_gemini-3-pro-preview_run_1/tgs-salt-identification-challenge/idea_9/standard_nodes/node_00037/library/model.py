import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Block.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Channel Squeeze and Excitation
        b, c, _, _ = x.size()
        cse = self.cSE(x).view(b, c, 1, 1)
        x_cse = x * cse

        # Spatial Squeeze and Excitation
        sse = self.sSE(x)
        x_sse = x * sse

        # Concurrent combination
        return x_cse + x_sse


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with optional downsampling.
    Structure: Conv3x3-BN-ReLU -> Conv3x3-BN -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """
    Decoder block with Bilinear Upsampling, Concatenation, and SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 1x1 Conv to reduce channels after concatenation
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.res_block = ResidualBlock(out_channels, out_channels)
        self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Ensure dimensions match (handling potential rounding issues)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Reduce and Refine
        x = self.conv_reduce(x)
        x = self.res_block(x)
        x = self.scse(x)
        return x


class DeepResUNet(nn.Module):
    """
    High-Capacity Deep Residual U-Net with Consistent Compound Loss and Snapshot Ensembling support.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(DeepResUNet, self).__init__()

        # Input: Image channels + 1 Depth channel
        self.actual_in_channels = in_channels + 1

        # --- Encoder ---
        # Level 1: 128x128 (No stride)
        self.enc1 = nn.Sequential(
            nn.Conv2d(
                self.actual_in_channels, 64, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Level 2: 64x64
        self.enc2 = ResidualBlock(64, 128, stride=2)
        # Level 3: 32x32
        self.enc3 = ResidualBlock(128, 256, stride=2)
        # Level 4: 16x16
        self.enc4 = ResidualBlock(256, 512, stride=2)
        # Level 5 (Bottleneck): 8x8
        self.enc5 = ResidualBlock(512, 512, stride=2)

        # --- Decoder ---
        # D4: 16x16. In: 512. Skip: 512. Out: 512
        self.dec4 = DecoderBlock(512, 512, 512)

        # D3: 32x32. In: 512. Skip: 256. Out: 256
        self.dec3 = DecoderBlock(512, 256, 256)

        # D2: 64x64. In: 256. Skip: 128. Out: 128
        self.dec2 = DecoderBlock(256, 128, 128)

        # D1: 128x128. In: 128. Skip: 64. Out: 64
        self.dec1 = DecoderBlock(128, 64, 64)

        # --- Deep Supervision Heads ---
        # Heads at 32x32 (from dec3), 64x64 (from dec2), 128x128 (from dec1)
        self.head32 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.head64 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.head128 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, depth_val):
        """
        Args:
            x: Input images [B, C, H, W] (typically 101x101)
            depth_val: Depth values [B] or [B, 1]
        """
        # 1. Preprocessing: Reflection Pad to 128x128
        # Target size 128
        target_h, target_w = 128, 128
        h, w = x.size(2), x.size(3)

        pad_h = target_h - h
        pad_w = target_w - w

        p_l = pad_w // 2
        p_r = pad_w - p_l
        p_t = pad_h // 2
        p_b = pad_h - p_t

        x_padded = F.pad(x, (p_l, p_r, p_t, p_b), mode="reflect")

        # 2. Fuse Depth
        b, _, h_p, w_p = x_padded.size()

        # Normalize depth (assuming raw feet/meters ~50-1000)
        # Using a simple scaling to 0-1 range roughly
        d_norm = depth_val.view(b, 1, 1, 1).float() / 1000.0
        depth_channel = d_norm.repeat(1, 1, h_p, w_p)

        x_in = torch.cat([x_padded, depth_channel], dim=1)

        # 3. Encoder
        e1 = self.enc1(x_in)  # 128x128, 64
        e2 = self.enc2(e1)  # 64x64, 128
        e3 = self.enc3(e2)  # 32x32, 256
        e4 = self.enc4(e3)  # 16x16, 512
        e5 = self.enc5(e4)  # 8x8, 1024

        # 4. Decoder with Deep Supervision
        d4 = self.dec4(e5, e4)  # 16x16, 512

        d3 = self.dec3(d4, e3)  # 32x32, 256
        out32 = self.head32(d3)

        d2 = self.dec2(d3, e2)  # 64x64, 128
        out64 = self.head64(d2)

        d1 = self.dec1(d2, e1)  # 128x128, 64
        out128 = self.head128(d1)

        # 5. Post-processing: Upsample Aux & Crop to original size

        # Upsample aux outputs to 128x128
        out32_up = F.interpolate(
            out32, size=(target_h, target_w), mode="bilinear", align_corners=True
        )
        out64_up = F.interpolate(
            out64, size=(target_h, target_w), mode="bilinear", align_corners=True
        )

        # Crop
        # Indices correspond to the padding applied
        h_end = target_h - p_b
        w_end = target_w - p_r

        final_out = out128[..., p_t:h_end, p_l:w_end]
        aux1 = out64_up[..., p_t:h_end, p_l:w_end]
        aux2 = out32_up[..., p_t:h_end, p_l:w_end]

        if self.training:
            return final_out, aux1, aux2
        else:
            return final_out
