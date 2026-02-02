import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_JOINTS,
    JOINTS_DIM,
    N_MFCC,
    HIDDEN_SIZE,
    SAWTOOTH_DILATIONS,
    KERNEL_SIZE,
    DROPOUT,
    NUM_CLASSES,
)


class Chomp1d(nn.Module):
    """
    Removes the last elements of a sequence to ensure causal convolution.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class GatedInput(nn.Module):
    """
    Feature-wise gating layer: x = x * sigmoid(Wx + b)
    Dynamically suppresses noisy sensor channels.
    """

    def __init__(self, input_dim):
        super(GatedInput, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        g = self.sigmoid(self.gate(x))
        return x * g


class GatedTemporalBlock(nn.Module):
    """
    WaveNet-style Gated Dilated Temporal Convolutional Block.
    Output = Tanh(Filter) * Sigmoid(Gate)
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(GatedTemporalBlock, self).__init__()

        # Combined convolution for filter and gate to save compute
        # Output channels = 2 * n_outputs (half for filter, half for gate)
        self.conv1 = nn.Conv1d(
            n_inputs,
            2 * n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        # 1x1 conv for residual connection integration
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection matching
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        out = self.conv1(x)
        out = self.chomp1(out)

        # Split into filter and gate
        filter_out, gate_out = out.chunk(2, dim=1)

        # Gated activation
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        out = self.dropout1(out)

        # 1x1 projection
        out = self.conv2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class SawtoothTCN(nn.Module):
    """
    TCN with Sawtooth Dilation Schedule.
    Used for refinement stages.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(SawtoothTCN, self).__init__()
        layers = []
        num_levels = len(SAWTOOTH_DILATIONS)

        # We use a fixed hidden size for the TCN stack
        hidden_dim = num_channels

        # Input projection: num_inputs (classes) -> hidden_dim
        self.input_proj = nn.Conv1d(num_inputs, hidden_dim, 1)

        for i in range(num_levels):
            dilation_size = SAWTOOTH_DILATIONS[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                GatedTemporalBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

        # Output projection: hidden_dim -> num_inputs (classes)
        self.output_proj = nn.Conv1d(hidden_dim, num_inputs, 1)

    def forward(self, x):
        # x: (Batch, Classes, Time)
        x_proj = self.input_proj(x)
        features = self.network(x_proj)
        correction = self.output_proj(features)
        return correction


class NRGSNet(nn.Module):
    """
    Normalized Residual Gated-Sawtooth Network.
    3-Stage Cascade:
    1. Gated Bi-GRU Encoder -> L1
    2. Normalized Residual TCN -> L2 = LogSoftmax(L1 + TCN(L1))
    3. Normalized Residual TCN -> L3 = LogSoftmax(L2 + TCN(L2))
    """

    def __init__(self):
        super(NRGSNet, self).__init__()

        # Calculate Input Dimension
        # Kinematics: 20 joints * 3 (Pos, Vel, Acc) * 3 coords = 180
        # Audio: 13 MFCC
        self.input_dim = (NUM_JOINTS * 3 * JOINTS_DIM) + N_MFCC

        # --- Stage 1: Gated High-Capacity Kinematic Encoder ---
        self.gated_input = GatedInput(self.input_dim)

        # Bi-GRU: Hidden size is per direction, so output is 2 * HIDDEN_SIZE
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )

        # Projection to classes
        self.stage1_fc = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)

        # --- Stage 2: Normalized Residual Sawtooth Refinement ---
        # Input: Log-probs (NUM_CLASSES channels)
        # Internal TCN width: 256 (same as GRU output capacity)
        self.stage2_tcn = SawtoothTCN(
            num_inputs=NUM_CLASSES,
            num_channels=HIDDEN_SIZE * 2,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        )

        # --- Stage 3: Independent Normalized Residual Refinement ---
        self.stage3_tcn = SawtoothTCN(
            num_inputs=NUM_CLASSES,
            num_channels=HIDDEN_SIZE * 2,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        )

    def forward(self, x):
        # x: (Batch, Time, Input_Dim)

        # --- Stage 1 ---
        # Gating
        x_gated = self.gated_input(x)

        # GRU Backbone
        gru_out, _ = self.gru(x_gated)

        # Initial Prediction
        logits_1 = self.stage1_fc(gru_out)  # (B, T, C)
        log_probs_1 = F.log_softmax(logits_1, dim=2)

        # --- Stage 2 ---
        # Permute for TCN: (B, T, C) -> (B, C, T)
        tcn_in_1 = log_probs_1.permute(0, 2, 1)

        # Predict additive correction to logits
        correction_1 = self.stage2_tcn(tcn_in_1)  # (B, C, T)

        # Normalized Residual: LogSoftmax(L1 + Correction)
        # We treat L1 as the base logits for the next stage
        logits_2 = tcn_in_1 + correction_1
        log_probs_2 = F.log_softmax(logits_2, dim=1)  # Softmax over channels

        # --- Stage 3 ---
        tcn_in_2 = log_probs_2  # (B, C, T)

        correction_2 = self.stage3_tcn(tcn_in_2)

        logits_3 = tcn_in_2 + correction_2
        log_probs_3 = F.log_softmax(logits_3, dim=1)

        # Permute back to (B, T, C) for output
        out_1 = log_probs_1
        out_2 = log_probs_2.permute(0, 2, 1)
        out_3 = log_probs_3.permute(0, 2, 1)

        return out_1, out_2, out_3
