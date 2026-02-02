import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Root-Centric Moderate-Capacity Encoder.
    Consists of a Bi-directional GRU followed by a Linear projection.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()

        # Bi-GRU: Hidden size is split between directions
        # Config.HIDDEN_DIM is total hidden size (192) -> 96 per direction
        hidden_per_dir = Config.HIDDEN_DIM // 2

        self.gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=hidden_per_dir,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(Config.DROPOUT_ENCODER)
        self.fc = nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, InputDim)
        Returns:
            logits: (Batch, Time, NumClasses)
        """
        # GRU Output: (Batch, Time, HiddenDim)
        out, _ = self.gru(x)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class TemporalBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block.
    Output = Activation(Conv(Input)) + Input (Residual)
    Activation = Tanh(Filter) * Sigmoid(Gate)
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        self.conv_filter = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv_gate = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv_filter.weight)
        nn.init.xavier_uniform_(self.conv_gate.weight)
        nn.init.zeros_(self.conv_filter.bias)
        nn.init.zeros_(self.conv_gate.bias)

    def forward(self, x):
        """
        Args:
            x: (Batch, Channels, Time)
        """
        f = torch.tanh(self.conv_filter(x))
        g = torch.sigmoid(self.conv_gate(x))
        out = f * g
        out = self.dropout(out)

        # Residual connection
        # Assuming n_inputs == n_outputs for this architecture
        return x + out


class TCNStage(nn.Module):
    """
    Refinement Stage using a stack of TemporalBlocks.
    """

    def __init__(self):
        super(TCNStage, self).__init__()

        layers = []
        num_channels = Config.NUM_CLASSES
        kernel_size = Config.TCN_KERNEL_SIZE
        dropout = Config.TCN_DROPOUT
        dilations = Config.TCN_DILATIONS

        for dilation in dilations:
            # Calculate padding for centered convolution (Non-Causal)
            # Output length should equal input length
            # Padding = (kernel_size - 1) * dilation / 2
            padding = (kernel_size - 1) * dilation // 2

            layers.append(
                TemporalBlock(
                    n_inputs=num_channels,
                    n_outputs=num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Classes) - Input Probabilities
        Returns:
            out: (Batch, Time, Classes) - Refined Logits
        """
        # Transpose for Conv1d: (Batch, Classes, Time)
        x = x.transpose(1, 2)

        out = self.network(x)

        # Transpose back: (Batch, Time, Classes)
        out = out.transpose(1, 2)
        return out


class RCMCN(nn.Module):
    """
    Root-Centric Moderate-Capacity Network (RC-MCN).
    Three-Stage Cascaded Network:
    1. Bi-GRU Encoder
    2. TCN Refinement 1
    3. TCN Refinement 2
    """

    def __init__(self):
        super(RCMCN, self).__init__()

        self.stage1 = BiGRUEncoder()
        self.stage2 = TCNStage()
        self.stage3 = TCNStage()

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, InputDim)
        Returns:
            (logits1, logits2, logits3): Tuple of outputs from each stage for Deep Supervision.
        """
        # --- Stage 1: Encoder ---
        logits1 = self.stage1(x)
        # Apply Softmax to create probabilities for the next stage
        probs1 = F.softmax(logits1, dim=2)

        # --- Stage 2: Refinement 1 ---
        # Input: Probabilities from Stage 1
        logits2 = self.stage2(probs1)
        probs2 = F.softmax(logits2, dim=2)

        # --- Stage 3: Refinement 2 ---
        # Input: Probabilities from Stage 2
        logits3 = self.stage3(probs2)

        return logits1, logits2, logits3
