import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiagonalScalingLayer(nn.Module):
    """
    Learnable Diagonal Scaling Interface.
    Applies a learnable scalar weight to each input channel: Y = X * w.
    Solves magnitude alignment (mm vs audio) without saturation.
    """

    def __init__(self, input_dim):
        super(DiagonalScalingLayer, self).__init__()
        # Initialize weights to 1.0 so initial pass is identity-like
        self.weights = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        # weights: (InputDim) -> Broadcasts to (Batch, Time, InputDim)
        return x * self.weights


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Moderate-Capacity Encoder.
    Bi-Directional GRU with projection to class probabilities.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(BiGRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional, so split hidden dim
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        out, _ = self.gru(x)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class DilatedResidualLayer(nn.Module):
    """
    Building block for the MSTCN.
    Dilated Conv1D -> Gated Activation -> 1x1 Conv -> Residual
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedResidualLayer, self).__init__()

        # Centered padding calculation for non-causal convolution
        # padding = (kernel_size - 1) * dilation / 2
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        out = self.conv_dilated(x)

        # Gated Activation Unit (WaveNet style)
        # We assume out_channels is consistent, so we apply tanh and sigmoid to the same output
        # Usually Gated TCN splits channels, but here we follow a simpler gated structure
        # often used in action segmentation: tanh(out) * sigmoid(out)
        out = torch.tanh(out) * torch.sigmoid(out)

        out = self.conv_1x1(out)
        out = self.dropout(out)

        # Residual connection
        return x + out


class MSTCNBlock(nn.Module):
    """
    Monotonic Non-Causal Refinement Block.
    Stack of Dilated Residual Layers with increasing dilation.
    """

    def __init__(self, num_classes, hidden_dim, kernel_size, dilations):
        super(MSTCNBlock, self).__init__()

        # Project class probabilities to hidden dimension
        self.conv_in = nn.Conv1d(num_classes, hidden_dim, 1)

        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                DilatedResidualLayer(hidden_dim, hidden_dim, kernel_size, dilation)
            )

        # Project back to class probabilities
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) -> Needs permute for Conv1d
        out = x.permute(0, 2, 1)  # (Batch, Channels, Time)

        out = self.conv_in(out)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)

        out = out.permute(0, 2, 1)  # (Batch, Time, NumClasses)
        return out


class LSMCN(nn.Module):
    """
    Linearly-Scaled Moderate-Capacity Network (LSM-CN).
    Three-Stage Cascaded Network:
    1. Scaling + BiGRU Encoder
    2. TCN Refinement Stage 1
    3. TCN Refinement Stage 2 (Independent weights)
    """

    def __init__(self):
        super(LSMCN, self).__init__()

        # --- Stage 1: Encoder ---
        self.scaling = DiagonalScalingLayer(Config.INPUT_DIM)

        self.encoder = BiGRUEncoder(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES,
            dropout=Config.DROPOUT,
        )

        # --- Stage 2: Refinement 1 ---
        # Internal hidden dim for TCN is typically 64 in MS-TCN literature
        tcn_hidden_dim = 64

        self.stage2 = MSTCNBlock(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=tcn_hidden_dim,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
        )

        # --- Stage 3: Refinement 2 ---
        # Independent weights, same structure
        self.stage3 = MSTCNBlock(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=tcn_hidden_dim,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, InputDim)
        Returns:
            p1, p2, p3: Class probabilities for each stage (Batch, Time, NumClasses)
        """
        # --- Stage 1 ---
        x_scaled = self.scaling(x)
        logits_1 = self.encoder(x_scaled)
        p1 = F.softmax(logits_1, dim=2)

        # --- Stage 2 ---
        # Input is strictly the probabilities from previous stage
        logits_2 = self.stage2(p1)
        p2 = F.softmax(logits_2, dim=2)

        # --- Stage 3 ---
        logits_3 = self.stage3(p2)
        p3 = F.softmax(logits_3, dim=2)

        return p1, p2, p3
