import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ASPP1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super(ASPP1D, self).__init__()
        self.modules_list = nn.ModuleList()

        # 1x1 Conv Branch
        self.modules_list.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Conv Branches
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

        # Global Average Pooling Branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection
        num_branches = len(self.modules_list) + 1
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * num_branches, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.modules_list:
            res.append(conv(x))

        # Global pooling branch
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=x.size(2), mode="linear", align_corners=True)
        res.append(gap)

        res = torch.cat(res, dim=1)
        return self.project(res)


class AttentionGate1D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate1D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        out = self.psi(psi)

        return x * out


class AttentionGatedResUNet1D(nn.Module):
    def __init__(self):
        super(AttentionGatedResUNet1D, self).__init__()

        self.n_features = Config.NUM_FEATURES
        self.n_classes = Config.NUM_CLASSES
        self.base_filters = Config.BASE_FILTERS

        # Encoder
        self.enc1 = ResidualBlock1D(self.n_features, self.base_filters)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ResidualBlock1D(self.base_filters, self.base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ResidualBlock1D(self.base_filters * 2, self.base_filters * 4)
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = ResidualBlock1D(self.base_filters * 4, self.base_filters * 8)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck (ASPP)
        self.bottleneck = ASPP1D(
            self.base_filters * 8, self.base_filters * 16, dilations=[6, 12, 18]
        )

        # Decoder
        # Dec4 (Scale 8)
        self.up4 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att4 = AttentionGate1D(
            F_g=self.base_filters * 16,
            F_l=self.base_filters * 8,
            F_int=self.base_filters * 4,
        )
        self.dec4 = ResidualBlock1D(
            self.base_filters * 16 + self.base_filters * 8, self.base_filters * 8
        )
        self.out4 = nn.Conv1d(self.base_filters * 8, self.n_classes, 1)

        # Dec3 (Scale 4)
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att3 = AttentionGate1D(
            F_g=self.base_filters * 8,
            F_l=self.base_filters * 4,
            F_int=self.base_filters * 2,
        )
        self.dec3 = ResidualBlock1D(
            self.base_filters * 8 + self.base_filters * 4, self.base_filters * 4
        )
        self.out3 = nn.Conv1d(self.base_filters * 4, self.n_classes, 1)

        # Dec2 (Scale 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att2 = AttentionGate1D(
            F_g=self.base_filters * 4,
            F_l=self.base_filters * 2,
            F_int=self.base_filters,
        )
        self.dec2 = ResidualBlock1D(
            self.base_filters * 4 + self.base_filters * 2, self.base_filters * 2
        )
        self.out2 = nn.Conv1d(self.base_filters * 2, self.n_classes, 1)

        # Dec1 (Scale 1 - Final)
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.att1 = AttentionGate1D(
            F_g=self.base_filters * 2,
            F_l=self.base_filters,
            F_int=self.base_filters // 2,
        )
        self.dec1 = ResidualBlock1D(
            self.base_filters * 2 + self.base_filters, self.base_filters
        )
        self.out1 = nn.Conv1d(self.base_filters, self.n_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder

        # Level 4
        d4_up = self.up4(b)
        # Attention Gate: g=d4_up, x=e4
        e4_gated = self.att4(g=d4_up, x=e4)
        d4_cat = torch.cat([d4_up, e4_gated], dim=1)
        d4 = self.dec4(d4_cat)
        out_scale8 = self.out4(d4)

        # Level 3
        d3_up = self.up3(d4)
        e3_gated = self.att3(g=d3_up, x=e3)
        d3_cat = torch.cat([d3_up, e3_gated], dim=1)
        d3 = self.dec3(d3_cat)
        out_scale4 = self.out3(d3)

        # Level 2
        d2_up = self.up2(d3)
        e2_gated = self.att2(g=d2_up, x=e2)
        d2_cat = torch.cat([d2_up, e2_gated], dim=1)
        d2 = self.dec2(d2_cat)
        out_scale2 = self.out2(d2)

        # Level 1
        d1_up = self.up1(d2)
        e1_gated = self.att1(g=d1_up, x=e1)
        d1_cat = torch.cat([d1_up, e1_gated], dim=1)
        d1 = self.dec1(d1_cat)
        out_scale1 = self.out1(d1)

        # Return list of outputs: [Scale 1 (Final), Scale 2, Scale 4, Scale 8]
        return [out_scale1, out_scale2, out_scale4, out_scale8]
