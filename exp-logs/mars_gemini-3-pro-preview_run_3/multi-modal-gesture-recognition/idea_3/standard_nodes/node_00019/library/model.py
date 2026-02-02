import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    GRU_HIDDEN_DIM,
    GRU_NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
    TCN_NUM_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
)


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-directional GRU.
    Extracts local temporal dynamics and generates initial frame-wise class logits.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if GRU_NUM_LAYERS > 1 else 0.0,
        )

        # Output layer: Maps from (Hidden * 2) to NumClasses
        self.fc = nn.Linear(GRU_HIDDEN_DIM * 2, NUM_CLASSES)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)

        # GRU Output: (Batch, Time, Hidden * 2)
        out, _ = self.gru(x)

        out = self.dropout(out)

        # Project to classes: (Batch, Time, NumClasses)
        logits = self.fc(out)

        return logits


class SingleStageTCN(nn.Module):
    """
    Stage 2: Temporal Convolutional Network for refinement.
    Takes frame-wise probabilities and refines them using temporal context.
    """

    def __init__(self, input_dim, num_classes, num_channels, kernel_size, dropout):
        super(SingleStageTCN, self).__init__()
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)

        # First layer: Project input to channel size
        self.conv_in = nn.Conv1d(input_dim, num_channels[0], 1)

        for i in range(num_levels):
            dilation = 2**i
            in_channels = num_channels[i]
            out_channels = num_channels[i]
            # Padding to maintain sequence length: (k-1) * d // 2
            padding = (kernel_size - 1) * dilation // 2

            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.conv_out = nn.Conv1d(num_channels[-1], num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        # Permute for Conv1d: (Batch, InputDim, Time)
        out = x.permute(0, 2, 1)

        out = self.conv_in(out)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Permute back: (Batch, Time, Classes)
        return out.permute(0, 2, 1)


class CascadedRefinementNet(nn.Module):
    """
    Two-stage architecture:
    1. BiGRU Encoder -> Logits
    2. TCN Refinement -> Refined Logits
    """

    def __init__(self):
        super(CascadedRefinementNet, self).__init__()
        self.stage1 = BiGRUEncoder()

        # Stage 2 takes Softmax probabilities from Stage 1
        # Input dim is NUM_CLASSES
        self.stage2 = SingleStageTCN(
            input_dim=NUM_CLASSES,
            num_classes=NUM_CLASSES,
            num_channels=TCN_NUM_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=TCN_DROPOUT,
        )

    def forward(self, x):
        # Stage 1
        s1_logits = self.stage1(x)

        # Convert logits to probabilities for Stage 2 input
        s1_probs = F.softmax(s1_logits, dim=2)

        # Stage 2
        s2_logits = self.stage2(s1_probs)

        return s1_logits, s2_logits
