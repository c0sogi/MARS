import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    TCN_LAYERS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
    TCN_CHANNELS,
    NUM_CLASSES,
    DILATIONS,
)


class GatedActivationUnit(nn.Module):
    """
    Gated Activation Unit for Temporal Convolutional Networks.
    Implements: Output = tanh(W_f * x) * sigmoid(W_g * x)
    Includes Dropout and Residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedActivationUnit, self).__init__()

        # Dilated convolution that outputs filter and gate maps simultaneously
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)

        # Handle residual connection dimension mismatch
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        out = self.conv(x)
        P, Q = out.chunk(2, dim=1)

        # Gated Activation
        out = torch.tanh(P) * torch.sigmoid(Q)
        out = self.dropout(out)
        out = self.conv_out(out)

        return res + out


class SingleStageTCN(nn.Module):
    """
    A single stage of the Multi-Stage TCN.
    Consists of a projection layer, a stack of GatedActivationUnits, and output heads.
    """

    def __init__(
        self,
        input_dim,
        num_layers,
        num_classes,
        hidden_dim,
        kernel_size,
        dropout,
        dilations,
    ):
        super(SingleStageTCN, self).__init__()

        # Initial projection to hidden dimension
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 2**i
            self.layers.append(
                GatedActivationUnit(
                    hidden_dim, hidden_dim, kernel_size, dilation, dropout
                )
            )

        # Output Heads
        self.conv_cls = nn.Conv1d(hidden_dim, num_classes, 1)
        self.conv_bnd = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x, mask):
        # x: (N, input_dim, L)
        # mask: (N, 1, L)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)
            # Apply mask to prevent noise propagation from padding
            if mask is not None:
                out = out * mask

        cls_logits = self.conv_cls(out)  # (N, num_classes, L)
        bnd_logits = self.conv_bnd(out)  # (N, 1, L)

        return cls_logits, bnd_logits


class GSG_CRCN(nn.Module):
    """
    Geometric Soft-Gated Cascaded Recurrent-Convolutional Network.
    Stage 1: Bi-LSTM Encoder
    Stage 2: Soft-Gated TCN Refinement
    Stage 3: Soft-Gated TCN Sharpening
    """

    def __init__(self):
        super(GSG_CRCN, self).__init__()

        # --- Stage 1: Geometric Recurrent Encoder ---
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0,
        )

        # Stage 1 Heads (Project from 2*HIDDEN_DIM)
        self.s1_cls = nn.Linear(HIDDEN_DIM * 2, NUM_CLASSES)
        self.s1_bnd = nn.Linear(HIDDEN_DIM * 2, 1)

        # --- Stage 2: Soft-Gated Refinement ---
        # Input: Concatenated Class Probs (21) + Boundary Prob (1) = 22
        self.stage2 = SingleStageTCN(
            input_dim=NUM_CLASSES + 1,
            num_layers=TCN_LAYERS,
            num_classes=NUM_CLASSES,
            hidden_dim=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=TCN_DROPOUT,
            dilations=DILATIONS,
        )

        # --- Stage 3: Soft-Gated Sharpening ---
        self.stage3 = SingleStageTCN(
            input_dim=NUM_CLASSES + 1,
            num_layers=TCN_LAYERS,
            num_classes=NUM_CLASSES,
            hidden_dim=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=TCN_DROPOUT,
            dilations=DILATIONS,
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input features (N, L, INPUT_DIM)
            mask: Sequence mask (N, L)
        Returns:
            dict: Logits for all stages.
        """
        # --- Stage 1 ---
        lstm_out, _ = self.lstm(x)  # (N, L, 2*HIDDEN)

        s1_cls_logits = self.s1_cls(lstm_out)  # (N, L, 21)
        s1_bnd_logits = self.s1_bnd(lstm_out)  # (N, L, 1)

        # Prepare input for Stage 2 (Probabilities)
        s1_probs = F.softmax(s1_cls_logits, dim=2)  # (N, L, 21)
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits)  # (N, L, 1)

        # Concatenate and Transpose for TCN (N, C, L)
        s1_out = torch.cat([s1_probs, s1_bnd_probs], dim=2)  # (N, L, 22)
        s1_out = s1_out.permute(0, 2, 1)  # (N, 22, L)

        # Expand mask for broadcasting (N, 1, L)
        mask_expanded = mask.unsqueeze(1)
        s1_out = s1_out * mask_expanded

        # --- Stage 2 ---
        s2_cls_logits, s2_bnd_logits = self.stage2(
            s1_out, mask_expanded
        )  # Logits: (N, 21, L), (N, 1, L)

        # Prepare input for Stage 3
        s2_probs = F.softmax(s2_cls_logits, dim=1)  # (N, 21, L)
        s2_bnd_probs = torch.sigmoid(s2_bnd_logits)  # (N, 1, L)

        s2_out = torch.cat([s2_probs, s2_bnd_probs], dim=1)  # (N, 22, L)
        s2_out = s2_out * mask_expanded

        # --- Stage 3 ---
        s3_cls_logits, s3_bnd_logits = self.stage3(s2_out, mask_expanded)

        # Return results
        # Permute TCN outputs back to (N, L, C) to match LSTM output format and standard CrossEntropy expectations
        return {
            "stage1": {
                "cls": s1_cls_logits,  # (N, L, 21)
                "bnd": s1_bnd_logits,  # (N, L, 1)
            },
            "stage2": {
                "cls": s2_cls_logits.permute(0, 2, 1),  # (N, L, 21)
                "bnd": s2_bnd_logits.permute(0, 2, 1),  # (N, L, 1)
            },
            "stage3": {
                "cls": s3_cls_logits.permute(0, 2, 1),  # (N, L, 21)
                "bnd": s3_bnd_logits.permute(0, 2, 1),  # (N, L, 1)
            },
        }
