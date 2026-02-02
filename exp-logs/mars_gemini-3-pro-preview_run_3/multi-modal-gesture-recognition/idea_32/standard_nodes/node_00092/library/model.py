import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class FeatureGating(nn.Module):
    """
    Learns a feature-wise gate to suppress noisy input channels dynamically.
    Formula: x = x * sigmoid(W * x + b)
    """

    def __init__(self, input_dim):
        super(FeatureGating, self).__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        gate = torch.sigmoid(self.gate_fc(x))
        return x * gate


class DilatedResidualLayer(nn.Module):
    """
    Gated Dilated Temporal Convolutional Layer with Residual Connection.
    Uses centered padding for non-causal context.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding for 'same' output length with centered alignment
        # For kernel_size=3, padding = dilation
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv_filter = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.conv_gate = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        # Gated Activation Unit
        out_filter = torch.tanh(self.conv_filter(x))
        out_gate = torch.sigmoid(self.conv_gate(x))
        out = out_filter * out_gate

        # Projection
        out = self.conv_1x1(out)
        out = self.dropout(out)

        # Residual connection
        return x + out


class MSTCNBlock(nn.Module):
    """
    Multi-Stage TCN Block consisting of stacked DilatedResidualLayers.
    """

    def __init__(
        self, num_layers, in_channels, hidden_channels, kernel_size, dilations, dropout
    ):
        super(MSTCNBlock, self).__init__()

        self.layers = nn.ModuleList()

        # First layer projects input to hidden_channels if necessary
        # But usually we project before entering the block or the block handles it.
        # Here we assume the block handles the full stack.

        for i in range(num_layers):
            dilation = dilations[i % len(dilations)]
            # First layer might have different input channels if we didn't project before
            # But to keep residual connections simple, we usually project input to hidden_channels first.
            self.layers.append(
                DilatedResidualLayer(
                    hidden_channels, hidden_channels, kernel_size, dilation, dropout
                )
            )

    def forward(self, x):
        # x shape: (Batch, HiddenChannels, Time)
        for layer in self.layers:
            x = layer(x)
        return x


class GHCMN(nn.Module):
    """
    Gated High-Capacity Monotonic Network.
    Stage 1: Gated Bi-GRU Encoder
    Stage 2: Monotonic TCN Refinement
    Stage 3: Independent Monotonic TCN Refinement
    """

    def __init__(self):
        super(GHCMN, self).__init__()

        # --- Dimensions ---
        # Skeleton: 20 joints * 9 channels (pos, vel, acc) = 180
        # Audio: 13 MFCCs
        self.input_dim = (config.SKELETON_JOINTS * config.SKELETON_CHANNELS) + 13
        self.num_classes = config.NUM_CLASSES
        self.hidden_dim = config.HIDDEN_DIM  # 256
        self.tcn_channels = config.TCN_CHANNELS  # 64

        # --- Stage 1: Encoder ---
        self.feature_gating = FeatureGating(self.input_dim)

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim // 2,  # Bidirectional, so half per direction
            num_layers=config.ENCODER_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.ENCODER_LAYERS > 1 else 0.0,
        )

        self.dropout = nn.Dropout(config.DROPOUT)
        self.stage1_fc = nn.Linear(self.hidden_dim, self.num_classes)

        # --- Stage 2: Refinement ---
        # Input to Stage 2 is the probabilities (or logits) from Stage 1.
        # Dimension: num_classes
        self.stage2_input_proj = nn.Conv1d(self.num_classes, self.tcn_channels, 1)

        self.stage2_tcn = MSTCNBlock(
            num_layers=len(config.TCN_DILATIONS),
            in_channels=self.tcn_channels,
            hidden_channels=self.tcn_channels,
            kernel_size=config.TCN_KERNEL_SIZE,
            dilations=config.TCN_DILATIONS,
            dropout=config.DROPOUT,
        )

        self.stage2_fc = nn.Conv1d(self.tcn_channels, self.num_classes, 1)

        # --- Stage 3: Refinement ---
        # Independent weights from Stage 2
        self.stage3_input_proj = nn.Conv1d(self.num_classes, self.tcn_channels, 1)

        self.stage3_tcn = MSTCNBlock(
            num_layers=len(config.TCN_DILATIONS),
            in_channels=self.tcn_channels,
            hidden_channels=self.tcn_channels,
            kernel_size=config.TCN_KERNEL_SIZE,
            dilations=config.TCN_DILATIONS,
            dropout=config.DROPOUT,
        )

        self.stage3_fc = nn.Conv1d(self.tcn_channels, self.num_classes, 1)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Features)
        Returns:
            dict containing outputs from all stages.
        """
        # --- Stage 1 ---
        # Gating
        x_gated = self.feature_gating(x)

        # GRU
        # self.gru returns (output, h_n)
        # output shape: (Batch, Time, HiddenDim)
        gru_out, _ = self.gru(x_gated)
        gru_out = self.dropout(gru_out)

        # Classification
        logits_1 = self.stage1_fc(gru_out)
        probs_1 = F.softmax(logits_1, dim=2)  # (Batch, Time, Classes)

        # --- Stage 2 ---
        # Prepare input for TCN: (Batch, Classes, Time)
        tcn_in_2 = probs_1.permute(0, 2, 1)

        # Project and Refine
        x2 = self.stage2_input_proj(tcn_in_2)
        x2 = self.stage2_tcn(x2)
        logits_2 = self.stage2_fc(x2)  # (Batch, Classes, Time)

        # Permute back for consistency
        logits_2 = logits_2.permute(0, 2, 1)  # (Batch, Time, Classes)
        probs_2 = F.softmax(logits_2, dim=2)

        # --- Stage 3 ---
        # Input is probs from Stage 2
        tcn_in_3 = probs_2.permute(0, 2, 1)

        # Project and Refine
        x3 = self.stage3_input_proj(tcn_in_3)
        x3 = self.stage3_tcn(x3)
        logits_3 = self.stage3_fc(x3)

        logits_3 = logits_3.permute(0, 2, 1)
        probs_3 = F.softmax(logits_3, dim=2)

        return {
            "stage1_logits": logits_1,
            "stage2_logits": logits_2,
            "stage3_logits": logits_3,
            "stage1_probs": probs_1,
            "stage2_probs": probs_2,
            "stage3_probs": probs_3,
        }
