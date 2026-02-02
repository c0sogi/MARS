import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledGating(nn.Module):
    """
    Decoupled-Norm Input Gating Mechanism.

    Logic:
    - Path A (Gate Generation): Input -> LayerNorm -> Linear -> Sigmoid -> Gate
    - Path B (Signal Retention): Input (Raw)
    - Output: Path B * Path A

    This preserves the physical magnitude of the raw signal (e.g., millimeters) while
    using normalized features to determine which parts of the signal are relevant.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: [Batch, Time, Dim]
        x_norm = self.norm(x)
        gate = torch.sigmoid(self.gate_fc(x_norm))
        return x * gate


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath) module.

    Randomly drops the entire residual branch during training with probability `prob`.
    This forces the network to rely on the identity connection, effectively training
    an ensemble of shallower networks.
    """

    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        # Bernoulli trial: drop entire tensor with probability self.prob
        if torch.rand(1).item() < self.prob:
            return torch.zeros_like(x)
        return x


class TemporalBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block with Stochastic Depth.

    Structure:
    - Conv1 (Dilated) -> ReLU -> Dropout
    - Conv2 (Dilated) -> ReLU -> Dropout
    - Residual Connection + Stochastic Depth
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        stochastic_prob=0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        self.relu = nn.ReLU()
        self.stochastic = StochasticDepth(stochastic_prob)

    def forward(self, x):
        # x: [Batch, Channels, Time]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        # Apply Stochastic Depth to the residual branch
        out = self.stochastic(out)

        return self.relu(out + res)


class DGCKN(nn.Module):
    """
    Stochastic-Regularized Decoupled-Gated Network (DGCKN).

    A Three-Stage Cascaded Network:
    1. Stage 1: Decoupled-Gated Bi-GRU Encoder (Raw Features -> P1)
    2. Stage 2: Stochastic-Depth Monotonic TCN Refinement (P1 -> P2)
    3. Stage 3: Independent Stochastic-Depth Refinement (P2 -> P3)
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()

        # ==========================
        # Stage 1: Encoder
        # ==========================
        self.gating = DecoupledGating(input_dim)
        self.gru = nn.GRU(
            input_dim,
            Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )
        # Bi-GRU output is Hidden*2
        self.fc1 = nn.Linear(Config.HIDDEN_DIM * 2, num_classes)

        # ==========================
        # Stage 2: Refinement
        # ==========================
        # Monotonically increasing dilation for large receptive field
        tcn_channels = [64, 64, 64, 64, 64]
        kernel_size = 3
        dilations = [1, 2, 4, 8, 16]

        self.tcn2 = nn.ModuleList()
        in_ch = num_classes
        for i, d in enumerate(dilations):
            out_ch = tcn_channels[i]
            padding = (kernel_size - 1) * d // 2
            self.tcn2.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=padding,
                    dropout=0.2,
                    stochastic_prob=Config.STOCHASTIC_DROP_PROB,
                )
            )
            in_ch = out_ch
        self.fc2 = nn.Linear(in_ch, num_classes)

        # ==========================
        # Stage 3: Independent Refinement
        # ==========================
        self.tcn3 = nn.ModuleList()
        in_ch = num_classes  # Input is P2 (num_classes)
        for i, d in enumerate(dilations):
            out_ch = tcn_channels[i]
            padding = (kernel_size - 1) * d // 2
            self.tcn3.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=padding,
                    dropout=0.2,
                    stochastic_prob=Config.STOCHASTIC_DROP_PROB,
                )
            )
            in_ch = out_ch
        self.fc3 = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: [Batch, Time, Dim]

        # --- Stage 1 Forward ---
        x_gated = self.gating(x)
        gru_out, _ = self.gru(x_gated)
        logits1 = self.fc1(gru_out)
        probs1 = torch.softmax(logits1, dim=2)  # [Batch, Time, Classes]

        # --- Stage 2 Forward ---
        # TCN expects [Batch, Channels, Time], so we transpose
        x2 = probs1.transpose(1, 2)
        for layer in self.tcn2:
            x2 = layer(x2)
        x2 = x2.transpose(1, 2)  # Back to [Batch, Time, Channels]
        logits2 = self.fc2(x2)
        probs2 = torch.softmax(logits2, dim=2)

        # --- Stage 3 Forward ---
        x3 = probs2.transpose(1, 2)
        for layer in self.tcn3:
            x3 = layer(x3)
        x3 = x3.transpose(1, 2)
        logits3 = self.fc3(x3)

        return logits1, logits2, logits3
