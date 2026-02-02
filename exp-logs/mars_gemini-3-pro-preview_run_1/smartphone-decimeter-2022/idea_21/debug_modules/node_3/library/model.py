import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 1, padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (1D) to capture multi-scale context.
    """

    def __init__(self, in_channels, out_channels, dilations):
        super(ASPP, self).__init__()
        modules = []

        # 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Convolutions
        for dilation in dilations:
            if dilation == 1:
                continue  # Skip 1, already covered by 1x1 or standard conv logic if desired
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.ModuleList(modules)

        # Projection layer
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * (len(modules) + 1), out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Global pooling branch
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=x.size(2), mode="nearest")
        res.append(gap)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ResUNet1D(nn.Module):
    """
    1D Residual U-Net with ASPP and Decimated Deep Supervision.
    """

    def __init__(self):
        super(ResUNet1D, self).__init__()

        self.in_channels = Config.IN_CHANNELS
        enc_dims = Config.ENCODER_CHANNELS  # e.g., [64, 128, 256, 512]
        dec_dims = Config.DECODER_CHANNELS  # e.g., [256, 128, 64, 32]

        # --- Encoder ---
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, enc_dims[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(enc_dims[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder Stage 1
        self.enc_block1 = ResBlock1D(enc_dims[0], enc_dims[0])
        self.pool1 = nn.MaxPool1d(2)

        # Encoder Stage 2
        self.enc_block2 = ResBlock1D(enc_dims[0], enc_dims[1])
        self.pool2 = nn.MaxPool1d(2)

        # Encoder Stage 3
        self.enc_block3 = ResBlock1D(enc_dims[1], enc_dims[2])
        self.pool3 = nn.MaxPool1d(2)

        # Encoder Stage 4
        self.enc_block4 = ResBlock1D(enc_dims[2], enc_dims[3])
        self.pool4 = nn.MaxPool1d(2)

        # --- Bottleneck ---
        self.aspp = ASPP(enc_dims[3], enc_dims[3], Config.ASPP_DILATIONS)

        # --- Decoder ---
        # Decoder Stage 1 (Processes L/16 features, upsamples to L/8)
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        # Input channels: Bottleneck (512) + Skip Enc4 (512) = 1024
        self.dec_block1 = ResBlock1D(enc_dims[3] + enc_dims[3], dec_dims[0])

        # Decoder Stage 2 (Upsamples to L/4)
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        # Input channels: Dec1 (256) + Skip Enc3 (256) = 512
        self.dec_block2 = ResBlock1D(dec_dims[0] + enc_dims[2], dec_dims[1])
        self.aux_head2 = nn.Conv1d(dec_dims[1], 2, 1)  # Aux Head at L/4 resolution

        # Decoder Stage 3 (Upsamples to L/2)
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        # Input channels: Dec2 (128) + Skip Enc2 (128) = 256
        self.dec_block3 = ResBlock1D(dec_dims[1] + enc_dims[1], dec_dims[2])
        self.aux_head3 = nn.Conv1d(dec_dims[2], 2, 1)  # Aux Head at L/2 resolution

        # Decoder Stage 4 (Upsamples to L)
        self.up4 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        # Input channels: Dec3 (64) + Skip Enc1 (64) = 128
        self.dec_block4 = ResBlock1D(dec_dims[2] + enc_dims[0], dec_dims[3])

        # Final Head
        self.final_head = nn.Conv1d(dec_dims[3], 2, 1)

    def forward(self, x):
        # x: (Batch, Channels, Length)

        # --- Encoder Path ---
        x = self.stem(x)

        e1 = self.enc_block1(x)  # (B, 64, L)
        p1 = self.pool1(e1)  # (B, 64, L/2)

        e2 = self.enc_block2(p1)  # (B, 128, L/2)
        p2 = self.pool2(e2)  # (B, 128, L/4)

        e3 = self.enc_block3(p2)  # (B, 256, L/4)
        p3 = self.pool3(e3)  # (B, 256, L/8)

        e4 = self.enc_block4(p3)  # (B, 512, L/8)
        p4 = self.pool4(e4)  # (B, 512, L/16)

        # --- Bottleneck ---
        b = self.aspp(p4)  # (B, 512, L/16)

        # --- Decoder Path ---
        # Stage 1
        d1 = self.up1(b)  # (B, 512, L/8)
        # Align size if input length was not power of 2
        if d1.size(2) != e4.size(2):
            d1 = F.interpolate(d1, size=e4.size(2), mode="linear", align_corners=True)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec_block1(d1)  # (B, 256, L/8)

        # Stage 2
        d2 = self.up2(d1)  # (B, 256, L/4)
        if d2.size(2) != e3.size(2):
            d2 = F.interpolate(d2, size=e3.size(2), mode="linear", align_corners=True)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec_block2(d2)  # (B, 128, L/4)
        aux2 = self.aux_head2(d2)  # (B, 2, L/4) -> Aux Output

        # Stage 3
        d3 = self.up3(d2)  # (B, 128, L/2)
        if d3.size(2) != e2.size(2):
            d3 = F.interpolate(d3, size=e2.size(2), mode="linear", align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec_block3(d3)  # (B, 64, L/2)
        aux3 = self.aux_head3(d3)  # (B, 2, L/2) -> Aux Output

        # Stage 4
        d4 = self.up4(d3)  # (B, 64, L)
        if d4.size(2) != e1.size(2):
            d4 = F.interpolate(d4, size=e1.size(2), mode="linear", align_corners=True)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec_block4(d4)  # (B, 32, L)

        # Final Output
        final = self.final_head(d4)  # (B, 2, L)

        # Return list: [Final, Aux_HighRes, Aux_LowRes]
        # Aux3 is L/2, Aux2 is L/4
        return [final, aux3, aux2]
