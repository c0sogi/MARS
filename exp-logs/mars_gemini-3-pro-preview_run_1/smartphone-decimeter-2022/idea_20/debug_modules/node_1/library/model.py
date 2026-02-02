import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CFG


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, dropout=0.0):
        super().__init__()
        self.modules_list = nn.ModuleList()

        # 1x1 Conv
        self.modules_list.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Convs
        for dilation in dilations:
            self.modules_list.append(
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

        # Global Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Project
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * (len(dilations) + 2), out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = []
        for mod in self.modules_list:
            res.append(mod(x))

        # Global pooling branch
        g = self.global_pool(x)
        g = F.interpolate(g, size=x.shape[-1], mode="linear", align_corners=False)
        res.append(g)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ResUNet1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = CFG.INPUT_DIM
        self.output_dim = CFG.OUTPUT_DIM
        enc_ch = CFG.ENCODER_CHANNELS
        dec_ch = CFG.DECODER_CHANNELS

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.input_dim, enc_ch[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(enc_ch[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc0 = ResidualBlock1D(enc_ch[0], enc_ch[0], dropout=CFG.DROPOUT_RATE)
        self.pool0 = nn.MaxPool1d(2)

        self.enc1 = ResidualBlock1D(enc_ch[0], enc_ch[1], dropout=CFG.DROPOUT_RATE)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ResidualBlock1D(enc_ch[1], enc_ch[2], dropout=CFG.DROPOUT_RATE)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ResidualBlock1D(enc_ch[2], enc_ch[3], dropout=CFG.DROPOUT_RATE)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck (ASPP)
        self.aspp = ASPP(
            enc_ch[3], dec_ch[0], CFG.ASPP_DILATIONS, dropout=CFG.DROPOUT_RATE
        )

        # Decoder
        # Dec0 (L/16 -> L/8)
        # Input: Bottleneck (256) + Skip3 (256) -> 512 -> ResBlock -> 256
        self.dec0_res = ResidualBlock1D(
            dec_ch[0] + enc_ch[3], dec_ch[0], dropout=CFG.DROPOUT_RATE
        )

        # Dec1 (L/8 -> L/4)
        # Input: Dec0 (256) -> Up -> 128 + Skip2 (128) -> 256 -> ResBlock -> 128
        self.dec1_up = nn.Conv1d(dec_ch[0], dec_ch[1], 1)
        self.dec1_res = ResidualBlock1D(
            dec_ch[1] + enc_ch[2], dec_ch[1], dropout=CFG.DROPOUT_RATE
        )

        # Dec2 (L/4 -> L/2)
        # Input: Dec1 (128) -> Up -> 64 + Skip1 (64) -> 128 -> ResBlock -> 64
        self.dec2_up = nn.Conv1d(dec_ch[1], dec_ch[2], 1)
        self.dec2_res = ResidualBlock1D(
            dec_ch[2] + enc_ch[1], dec_ch[2], dropout=CFG.DROPOUT_RATE
        )

        # Dec3 (L/2 -> L)
        # Input: Dec2 (64) -> Up -> 32 + Skip0 (32) -> 64 -> ResBlock -> 32
        self.dec3_up = nn.Conv1d(dec_ch[2], dec_ch[3], 1)
        self.dec3_res = ResidualBlock1D(
            dec_ch[3] + enc_ch[0], dec_ch[3], dropout=CFG.DROPOUT_RATE
        )

        # Heads
        self.head_main = nn.Conv1d(dec_ch[3], self.output_dim, 1)
        self.head_aux1 = nn.Conv1d(dec_ch[2], self.output_dim, 1)  # Off Dec2 (1/2 res)
        self.head_aux2 = nn.Conv1d(dec_ch[1], self.output_dim, 1)  # Off Dec1 (1/4 res)

    def forward(self, x):
        # x: (B, C, L)

        # Stem
        x = self.stem(x)

        # Encoder
        s0 = self.enc0(x)  # (B, 32, L)
        p0 = self.pool0(s0)  # (B, 32, L/2)

        s1 = self.enc1(p0)  # (B, 64, L/2)
        p1 = self.pool1(s1)  # (B, 64, L/4)

        s2 = self.enc2(p1)  # (B, 128, L/4)
        p2 = self.pool2(s2)  # (B, 128, L/8)

        s3 = self.enc3(p2)  # (B, 256, L/8)
        p3 = self.pool3(s3)  # (B, 256, L/16)

        # Bottleneck
        b = self.aspp(p3)  # (B, 256, L/16)

        # Decoder

        # Dec0 (L/16 -> L/8)
        d0 = F.interpolate(b, scale_factor=2, mode="linear", align_corners=False)
        d0 = torch.cat([d0, s3], dim=1)
        d0 = self.dec0_res(d0)

        # Dec1 (L/8 -> L/4)
        d1 = F.interpolate(d0, scale_factor=2, mode="linear", align_corners=False)
        d1 = self.dec1_up(d1)  # 256 -> 128
        d1 = torch.cat([d1, s2], dim=1)
        d1 = self.dec1_res(d1)

        # Aux2 Head (L/4)
        aux2 = self.head_aux2(d1)

        # Dec2 (L/4 -> L/2)
        d2 = F.interpolate(d1, scale_factor=2, mode="linear", align_corners=False)
        d2 = self.dec2_up(d2)  # 128 -> 64
        d2 = torch.cat([d2, s1], dim=1)
        d2 = self.dec2_res(d2)

        # Aux1 Head (L/2)
        aux1 = self.head_aux1(d2)

        # Dec3 (L/2 -> L)
        d3 = F.interpolate(d2, scale_factor=2, mode="linear", align_corners=False)
        d3 = self.dec3_up(d3)  # 64 -> 32
        d3 = torch.cat([d3, s0], dim=1)
        d3 = self.dec3_res(d3)

        # Main Head (L)
        main = self.head_main(d3)

        return [main, aux1, aux2]
