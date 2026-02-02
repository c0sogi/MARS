import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridSE(nn.Module):
    """
    Hybrid Squeeze-and-Excitation Module.
    Uses Global Average Pooling for the squeeze operation to act as a low-pass filter.
    """

    def __init__(self, channels, reduction=16):
        super(HybridSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class PlainBlock(nn.Module):
    """
    Standard CNN Block: Conv -> BN -> LeakyReLU -> SE -> MaxPool.
    Explicitly retains bias in Conv2d.
    """

    def __init__(self, in_channels, out_channels):
        super(PlainBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=Config.LEAKY_RELU_SLOPE, inplace=True)
        self.se = HybridSE(out_channels, reduction=Config.SE_REDUCTION)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class IsomorphicReadout(nn.Module):
    """
    Decoupled Isomorphic Readout.
    Processes features from Stage 3 and Stage 4 using separate projections
    and extracts both Max and Min statistics.
    """

    def __init__(self, in_channels_list):
        super(IsomorphicReadout, self).__init__()
        # Stage 3 is index 2, Stage 4 is index 3 in BACKBONE_CHANNELS
        c3 = in_channels_list[2]
        c4 = in_channels_list[3]

        # Target dimension for projection (128 -> 64)
        out_dim = 64

        self.proj3 = nn.Conv2d(c3, out_dim, kernel_size=1, bias=True)
        self.proj4 = nn.Conv2d(c4, out_dim, kernel_size=1, bias=True)

        self.output_dim = out_dim * 4  # (Max3 + Min3 + Max4 + Min4)

    def forward(self, x3, x4):
        # Project
        p3 = self.proj3(x3)
        p4 = self.proj4(x4)

        # Global Max Pooling
        max3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        max4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)

        # Global Min Pooling (implemented via negative max pool)
        min3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)
        min4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)

        # Concatenate: 64*4 = 256 dimensions
        return torch.cat([max3, min3, max4, min4], dim=1)


class MultiSampleDropoutHead(nn.Module):
    """
    Interaction-Aware Multi-Sample Dropout Head.
    Fuses image features with incidence angle, applies a non-linear interaction layer,
    then predicts via multiple dropout branches.
    """

    def __init__(self, input_dim, hidden_dim, num_samples, dropout_rate):
        super(MultiSampleDropoutHead, self).__init__()

        # Interaction Layer: Fuses 256 (Image) + 1 (Angle) -> Hidden
        self.interaction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=Config.LEAKY_RELU_SLOPE, inplace=True),
        )

        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

        # Parallel classification branches
        # We use a ModuleList of Linear layers. Dropout is functional or a separate module.
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_samples)]
        )

        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x is the fused vector
        features = self.interaction(x)

        logits_list = []
        for classifier in self.classifiers:
            # Apply dropout independently for each branch
            dropped = self.drop(features)
            logits_list.append(classifier(dropped))

        # Stack to shape (Batch, Num_Samples)
        return torch.cat(logits_list, dim=1)


class IAMSI_CNN(nn.Module):
    """
    Interaction-Aware Multi-Sample Isomorphic CNN.
    """

    def __init__(self):
        super(IAMSI_CNN, self).__init__()

        channels = Config.BACKBONE_CHANNELS

        # Backbone: 4 Stages
        self.block1 = PlainBlock(Config.IN_CHANNELS, channels[0])
        self.block2 = PlainBlock(channels[0], channels[1])
        self.block3 = PlainBlock(channels[1], channels[2])
        self.block4 = PlainBlock(channels[2], channels[3])

        # Readout
        self.readout = IsomorphicReadout(channels)

        # Head
        # Input dim = Readout output (256) + Angle (1)
        head_input_dim = self.readout.output_dim + 1
        self.head = MultiSampleDropoutHead(
            input_dim=head_input_dim,
            hidden_dim=Config.INTERACTION_HIDDEN_DIM,
            num_samples=Config.NUM_DROPOUT_SAMPLES,
            dropout_rate=Config.DROPOUT_RATE,
        )

        # Initialization (PyTorch default is Kaiming Uniform for Linear/Conv)
        # We rely on default initialization as per strategy.

    def forward(self, x, angle):
        # Backbone
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)  # Stage 3 features
        x4 = self.block4(x3)  # Stage 4 features

        # Readout
        img_features = self.readout(x3, x4)

        # Fusion
        # Ensure angle is (B, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Concatenate raw angle (no normalization)
        fused = torch.cat([img_features, angle], dim=1)

        # Head
        # Returns (Batch, Num_Samples)
        logits = self.head(fused)

        if self.training:
            # Return all logits for multi-sample loss calculation
            return logits
        else:
            # Inference: Average logits across branches
            return torch.mean(logits, dim=1, keepdim=True)
