import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_JOINTS,
    AUDIO_N_MFCC,
    HIDDEN_SIZE,
    DROPOUT_ENCODER,
    TCN_CHANNELS,
    TCN_DILATIONS,
    TCN_KERNEL_SIZE,
    DROPOUT_TCN,
    NUM_CLASSES,
)


class InputGating(nn.Module):
    """
    Learnable gating layer that dynamically scales input features.
    Formula: x_tilde = x * sigmoid(W * x + b)
    """

    def __init__(self, input_dim):
        super(InputGating, self).__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        gate = self.sigmoid(self.gate_fc(x))
        return x * gate


class GatedTCNBlock(nn.Module):
    """
    Dilated Temporal Convolutional Block with Gated Linear Unit (GLU) activation.
    Uses centered padding to maintain temporal resolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedTCNBlock, self).__init__()
        # For kernel_size=3, padding=dilation ensures centered convolution
        # L_out = L_in + 2*padding - dilation*(kernel_size-1) - 1 + 1
        # L_out = L_in + 2*d - d*(2) = L_in
        self.padding = dilation

        # Convolution produces 2 * out_channels for GLU (split into val and gate)
        self.conv = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

        # Residual connection projection if dimensions mismatch
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x

        out = self.conv(x)
        # Split for GLU: tanh(val) * sigmoid(gate)
        val, gate = torch.chunk(out, 2, dim=1)
        out = torch.tanh(val) * torch.sigmoid(gate)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class RefinementStage(nn.Module):
    """
    A stack of Gated TCN blocks for refining probability sequences.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dilations, dropout):
        super(RefinementStage, self).__init__()
        layers = []

        # First layer projects input (probabilities) to hidden dimension
        layers.append(
            GatedTCNBlock(input_dim, hidden_dim, TCN_KERNEL_SIZE, dilations[0], dropout)
        )

        # Subsequent layers with increasing dilation
        for d in dilations[1:]:
            layers.append(
                GatedTCNBlock(hidden_dim, hidden_dim, TCN_KERNEL_SIZE, d, dropout)
            )

        self.layers = nn.ModuleList(layers)
        # Final projection back to class space
        self.final_conv = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):
        # Input x: (Batch, Time, Classes)
        # Permute to (Batch, Classes, Time) for Conv1d
        x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x)

        out = self.final_conv(x)
        # Permute back to (Batch, Time, Classes)
        return out.permute(0, 2, 1)


class RGHCMN(nn.Module):
    """
    Regularized Gated High-Capacity Monotonic Network.
    Stage 1: Input Gating -> Bi-GRU Encoder
    Stage 2: Gated TCN Refinement (Monotonic Dilation)
    Stage 3: Independent Gated TCN Refinement
    """

    def __init__(self):
        super(RGHCMN, self).__init__()

        # Calculate Input Dimension
        # Kinematics: 20 joints * 9 features (Pos(3) + Vel(3) + Acc(3))
        # Audio: 13 MFCCs
        self.input_dim = (NUM_JOINTS * 9) + AUDIO_N_MFCC

        # --- Stage 1: Regularized Gated Encoder ---
        self.input_gating = InputGating(self.input_dim)

        # Bi-directional GRU
        # hidden_size is per direction, so output features = 2 * HIDDEN_SIZE
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_encoder = nn.Dropout(DROPOUT_ENCODER)
        self.fc_stage1 = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)

        # --- Stage 2: Monotonic Gated Refinement ---
        self.stage2 = RefinementStage(
            input_dim=NUM_CLASSES,
            hidden_dim=TCN_CHANNELS,
            num_classes=NUM_CLASSES,
            dilations=TCN_DILATIONS,
            dropout=DROPOUT_TCN,
        )

        # --- Stage 3: Independent Iterative Refinement ---
        self.stage3 = RefinementStage(
            input_dim=NUM_CLASSES,
            hidden_dim=TCN_CHANNELS,
            num_classes=NUM_CLASSES,
            dilations=TCN_DILATIONS,
            dropout=DROPOUT_TCN,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, InputDim)

        Returns:
            dict: Logits from all three stages for Deep Supervision.
        """
        # --- Stage 1 ---
        x_gated = self.input_gating(x)
        gru_out, _ = self.gru(x_gated)
        gru_out = self.dropout_encoder(gru_out)
        logits_1 = self.fc_stage1(gru_out)

        # Softmax for input to next stage (detach not strictly necessary unless stopping gradients,
        # but here we want gradients to flow back through probabilities)
        probs_1 = F.softmax(logits_1, dim=2)

        # --- Stage 2 ---
        logits_2 = self.stage2(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # --- Stage 3 ---
        logits_3 = self.stage3(probs_2)

        return {
            "logits_1": logits_1,
            "logits_2": logits_2,
            "logits_3": logits_3,
        }
