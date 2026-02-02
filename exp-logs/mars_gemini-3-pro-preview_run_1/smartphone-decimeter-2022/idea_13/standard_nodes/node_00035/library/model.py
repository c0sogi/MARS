import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolution layers and a skip connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ASPPModule1D(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for 1D sequences.
    Captures multi-scale temporal context.
    """

    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
        self.branches = nn.ModuleList()

        # Branch 1: 1x1 Conv
        self.branches.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Branch 2-N: Dilated Convs
        for d in dilations:
            # We use kernel=3 for dilated branches
            padding = d
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        3,
                        padding=padding,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Branch N+1: Global Average Pooling
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Project concatenated branches
        # Number of branches = 1 (1x1) + len(dilations) + 1 (global)
        total_channels = out_channels * (len(dilations) + 2)

        self.project = nn.Sequential(
            nn.Conv1d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []

        # Convolutional branches
        for branch in self.branches:
            res.append(branch(x))

        # Global pooling branch
        global_feat = self.global_branch(x)
        # Upsample global features to match input length
        global_feat = F.interpolate(
            global_feat, size=x.shape[2], mode="linear", align_corners=False
        )
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class MultiScaleResUNet1D(nn.Module):
    """
    U-Net architecture with Residual Blocks, ASPP bottleneck, and Deep Supervision.
    """

    def __init__(self):
        super().__init__()
        num_features = Config.NUM_FEATURES
        num_classes = Config.NUM_CLASSES
        base_filters = Config.NUM_FILTERS
        dilations = Config.ASPP_DILATIONS

        # --- Encoder ---
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, base_filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Stage 1
        self.enc1 = ResidualBlock1D(base_filters, base_filters * 2)
        self.pool1 = nn.MaxPool1d(2)

        # Stage 2
        self.enc2 = ResidualBlock1D(base_filters * 2, base_filters * 4)
        self.pool2 = nn.MaxPool1d(2)

        # Stage 3
        self.enc3 = ResidualBlock1D(base_filters * 4, base_filters * 8)
        self.pool3 = nn.MaxPool1d(2)

        # Stage 4
        self.enc4 = ResidualBlock1D(base_filters * 8, base_filters * 16)
        self.pool4 = nn.MaxPool1d(2)

        # --- Bottleneck ---
        self.aspp = ASPPModule1D(base_filters * 16, base_filters * 16, dilations)

        # --- Decoder ---
        # Stage 4 Up
        self.up4 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec4_reduce = nn.Conv1d(
            base_filters * 16 + base_filters * 16, base_filters * 8, 1, bias=False
        )
        self.dec4 = ResidualBlock1D(base_filters * 8, base_filters * 8)

        # Stage 3 Up
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec3_reduce = nn.Conv1d(
            base_filters * 8 + base_filters * 8, base_filters * 4, 1, bias=False
        )
        self.dec3 = ResidualBlock1D(base_filters * 4, base_filters * 4)

        # Stage 2 Up
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec2_reduce = nn.Conv1d(
            base_filters * 4 + base_filters * 4, base_filters * 2, 1, bias=False
        )
        self.dec2 = ResidualBlock1D(base_filters * 2, base_filters * 2)

        # Stage 1 Up
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec1_reduce = nn.Conv1d(
            base_filters * 2 + base_filters * 2, base_filters, 1, bias=False
        )
        self.dec1 = ResidualBlock1D(base_filters, base_filters)

        # --- Heads ---
        self.final_conv = nn.Conv1d(base_filters, num_classes, 1)

        # Auxiliary heads for Deep Supervision
        self.aux2 = nn.Conv1d(base_filters * 2, num_classes, 1)
        self.aux3 = nn.Conv1d(base_filters * 4, num_classes, 1)

    def forward(self, x):
        # x shape: (Batch, Features, Time)

        # Encoder Path
        x = self.stem(x)  # C: base

        e1 = self.enc1(x)  # C: base*2
        p1 = self.pool1(e1)  # T/2

        e2 = self.enc2(p1)  # C: base*4
        p2 = self.pool2(e2)  # T/4

        e3 = self.enc3(p2)  # C: base*8
        p3 = self.pool3(e3)  # T/8

        e4 = self.enc4(p3)  # C: base*16
        p4 = self.pool4(e4)  # T/16

        # Bottleneck
        b = self.aspp(p4)  # C: base*16, T/16

        # Decoder Path
        # Up 4
        d4 = self.up4(b)
        # Fix potential size mismatch from odd input dimensions
        if d4.shape[2] != e4.shape[2]:
            d4 = F.interpolate(d4, size=e4.shape[2], mode="linear", align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4_reduce(d4)
        d4 = self.dec4(d4)

        # Up 3
        d3 = self.up3(d4)
        if d3.shape[2] != e3.shape[2]:
            d3 = F.interpolate(d3, size=e3.shape[2], mode="linear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_reduce(d3)
        d3 = self.dec3(d3)

        # Aux Head 3
        aux3_out = self.aux3(d3)

        # Up 2
        d2 = self.up2(d3)
        if d2.shape[2] != e2.shape[2]:
            d2 = F.interpolate(d2, size=e2.shape[2], mode="linear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_reduce(d2)
        d2 = self.dec2(d2)

        # Aux Head 2
        aux2_out = self.aux2(d2)

        # Up 1
        d1 = self.up1(d2)
        if d1.shape[2] != e1.shape[2]:
            d1 = F.interpolate(d1, size=e1.shape[2], mode="linear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_reduce(d1)
        d1 = self.dec1(d1)

        # Final Output
        final_out = self.final_conv(d1)

        if self.training:
            # Upsample aux outputs to match final resolution for loss calculation
            aux3_out = F.interpolate(
                aux3_out, size=final_out.shape[2], mode="linear", align_corners=False
            )
            aux2_out = F.interpolate(
                aux2_out, size=final_out.shape[2], mode="linear", align_corners=False
            )
            return [final_out, aux2_out, aux3_out]
        else:
            return final_out
