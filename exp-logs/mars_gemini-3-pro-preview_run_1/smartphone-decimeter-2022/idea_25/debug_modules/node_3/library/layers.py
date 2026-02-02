import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolution layers and a skip connection.
    Structure: Conv1d -> BN -> ReLU -> Conv1d -> BN -> (+ Residual) -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(ResidualBlock1D, self).__init__()

        # Calculate padding to maintain sequence length (if stride=1)
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
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
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
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
    Atrous Spatial Pyramid Pooling (1D).
    Captures multi-scale context using parallel branches with different dilation rates.
    """

    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()

        self.branches = nn.ModuleList()

        # Branch 1: 1x1 Convolution
        self.branches.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Branches 2..N: Dilated 3x3 Convolutions
        for d in dilations:
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
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Project concatenated features to output channels
        # Total branches = 1 (1x1) + len(dilations) + 1 (pool)
        concat_channels = out_channels * (len(dilations) + 2)

        self.project = nn.Sequential(
            nn.Conv1d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []

        # Process conv branches
        for branch in self.branches:
            res.append(branch(x))

        # Process global pooling branch
        gap = self.global_avg_pool(x)
        # Upsample to match input length
        gap = F.interpolate(gap, size=x.size(2), mode="linear", align_corners=False)
        res.append(gap)

        # Concatenate and project
        res = torch.cat(res, dim=1)
        return self.project(res)


class AttentionGate1D(nn.Module):
    """
    1D Attention Gate.
    Uses a gating signal (g) to filter features from a skip connection (x).
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in gating signal (from decoder).
            F_l: Number of channels in skip connection (from encoder).
            F_int: Number of intermediate channels.
        """
        super(AttentionGate1D, self).__init__()

        # W_g: Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        # W_x: Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int),
        )

        # Psi: Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal (coarser scale features from decoder).
            x: Skip connection (finer scale features from encoder).
        Returns:
            Filtered x.
        """
        # Upsample g to match x size if necessary
        # Usually g is upsampled in the U-Net before this block, but we ensure alignment here.
        if g.size(2) != x.size(2):
            g = F.interpolate(g, size=x.size(2), mode="linear", align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Additive attention
        psi = self.relu(g1 + x1)

        # Compute attention map (0 to 1)
        alpha = self.psi(psi)

        # Weight the skip connection
        return x * alpha
