import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    KERNEL_SIZE,
    DROPOUT,
    DILATIONS,
)


class FeatureGatingLayer(nn.Module):
    """
    Applies a learnable frame-wise gating mechanism to the input features.
    Formula: Output = Input * Sigmoid(Linear(Input))
    """

    def __init__(self, input_dim):
        super(FeatureGatingLayer, self).__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        gates = self.sigmoid(self.gate_fc(x))
        return x * gates


class SeparableGatedConvBlock(nn.Module):
    """
    A parameter-efficient dilated convolution block using Depthwise Separable Convolutions
    and Gated Activations (WaveNet style).
    Structure:
        Input -> Depthwise Conv (Dilated) -> Pointwise Conv (2x Channels)
              -> Split (Content, Gate) -> Tanh(Content) * Sigmoid(Gate)
              -> Dropout -> Residual + Input
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(SeparableGatedConvBlock, self).__init__()

        # Padding to maintain sequence length (assuming odd kernel_size)
        padding = (kernel_size - 1) * dilation // 2

        # 1. Depthwise Convolution: Captures temporal context per channel
        # groups=in_channels makes it depthwise
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )

        # 2. Pointwise Convolution: Projects to 2x channels for Gated Activation
        # Maps in_channels -> 2 * out_channels
        self.pointwise = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Residual connection handling
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x

        # Depthwise
        out = self.depthwise(x)

        # Pointwise
        out = self.pointwise(out)

        # Gated Activation Unit
        # Split into Content (P) and Gate (Q)
        P, Q = out.chunk(2, dim=1)
        out = torch.tanh(P) * torch.sigmoid(Q)

        # Dropout
        out = self.dropout(out)

        # Residual Addition
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return out + residual


class RefinementStage(nn.Module):
    """
    A single refinement stage consisting of a projection layer,
    a stack of SeparableGatedConvBlocks, and a classification head.
    """

    def __init__(self, num_classes, hidden_dim, kernel_size, dilations, dropout):
        super(RefinementStage, self).__init__()

        # Project probabilities to hidden dimension
        self.input_proj = nn.Conv1d(num_classes, hidden_dim, kernel_size=1)

        # Stack of TCN blocks
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                SeparableGatedConvBlock(
                    hidden_dim, hidden_dim, kernel_size, dilation, dropout
                )
            )

        # Classifier
        self.output_proj = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, probs):
        # probs: (Batch, Time, NumClasses)
        # Transpose for Conv1d: (Batch, NumClasses, Time)
        x = probs.transpose(1, 2)

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_proj(x)

        # Transpose back: (Batch, Time, NumClasses)
        return x.transpose(1, 2)


class LGKRN(nn.Module):
    """
    Lightweight Gated-Kinematic Refinement Network.
    Stage 1: Gated Bi-GRU for initial prediction.
    Stage 2 & 3: Separable Gated TCNs for iterative refinement.
    """

    def __init__(self):
        super(LGKRN, self).__init__()

        # --- Stage 1: Gated Kinematic Sequence Encoder ---
        self.gating = FeatureGatingLayer(INPUT_DIM)

        # Bi-GRU Backbone
        # hidden_dim // 2 per direction -> concatenated to hidden_dim
        self.gru = nn.GRU(
            INPUT_DIM,
            HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.stage1_dropout = nn.Dropout(DROPOUT)
        self.stage1_fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

        # --- Stage 2: Parameter-Efficient Gated Refinement ---
        self.stage2 = RefinementStage(
            NUM_CLASSES, HIDDEN_DIM, KERNEL_SIZE, DILATIONS, DROPOUT
        )

        # --- Stage 3: Independent Iterative Refinement ---
        self.stage3 = RefinementStage(
            NUM_CLASSES, HIDDEN_DIM, KERNEL_SIZE, DILATIONS, DROPOUT
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Time, InputDim)

        Returns:
            tuple: (logits_1, logits_2, logits_3)
        """
        # --- Stage 1 ---
        # Feature Gating
        x_gated = self.gating(x)

        # Recurrent Encoder
        gru_out, _ = self.gru(x_gated)
        gru_out = self.stage1_dropout(gru_out)

        # Initial Prediction
        logits_1 = self.stage1_fc(gru_out)  # (Batch, Time, Classes)

        # Softmax for next stage input
        probs_1 = F.softmax(logits_1, dim=2)

        # --- Stage 2 ---
        logits_2 = self.stage2(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # --- Stage 3 ---
        logits_3 = self.stage3(probs_2)

        return logits_1, logits_2, logits_3
