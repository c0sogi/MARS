import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedDilatedConv1d(nn.Module):
    """
    A non-causal gated dilated temporal convolution layer.
    Implements: Tanh(W_f * x) * Sigmoid(W_g * x)
    Uses centered padding to ensure non-causality.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(GatedDilatedConv1d, self).__init__()

        # For kernel_size=3, centered padding = dilation
        # (k-1)*d / 2 = (2*d)/2 = d
        self.padding = dilation
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Filter convolution (Tanh)
        self.conv_f = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # Gate convolution (Sigmoid)
        self.conv_g = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.conv_res = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.conv_res = None

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        f = self.conv_f(x)
        g = self.conv_g(x)

        # Gated activation
        h = torch.tanh(f) * torch.sigmoid(g)
        h = self.dropout(h)

        # Residual
        if self.conv_res is not None:
            res = self.conv_res(x)
        else:
            res = x

        return h + res


class SingleStageTCN(nn.Module):
    """
    A single refinement stage consisting of stacked GatedDilatedConv1d layers
    following a sawtooth dilation schedule.
    """

    def __init__(self, num_classes, num_layers, hidden_channels, dropout):
        super(SingleStageTCN, self).__init__()

        # Input projection: NumClasses -> Hidden
        self.conv_in = nn.Conv1d(num_classes, hidden_channels, 1)

        # Stack layers based on sawtooth schedule
        layers = []
        dilations = Config.SAWTOOTH_DILATIONS

        # Ensure we have enough dilations defined for the requested layers
        # If num_layers > len(dilations), we cycle or extend.
        # Here we strictly follow the config schedule length usually.
        # But for flexibility, we iterate num_layers.

        for i in range(len(dilations)):
            dilation = dilations[i]
            layers.append(
                GatedDilatedConv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.layers = nn.ModuleList(layers)

        # Output projection: Hidden -> NumClasses
        self.conv_out = nn.Conv1d(hidden_channels, num_classes, 1)

    def forward(self, x):
        # x shape: (Batch, Time, NumClasses) -> Needs (Batch, NumClasses, Time) for Conv1d
        x = x.permute(0, 2, 1)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Back to (Batch, Time, NumClasses)
        out = out.permute(0, 2, 1)
        return out


class BiGRUEncoder(nn.Module):
    """
    Stage 1: High-Capacity Kinematic Sequence Encoder.
    """

    def __init__(self, input_dim, hidden_size, num_classes):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Project from 2*hidden to num_classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)

        # out shape: (Batch, Time, 2*Hidden)
        logits = self.fc(out)

        return logits


class HCNCSN(nn.Module):
    """
    High-Capacity Non-Causal Sawtooth Network.
    Stage 1: BiGRU Encoder
    Stage 2: TCN Refinement (Sawtooth)
    Stage 3: TCN Refinement (Sawtooth)
    """

    def __init__(self):
        super(HCNCSN, self).__init__()

        # Stage 1
        self.stage1 = BiGRUEncoder(
            input_dim=Config.INPUT_DIM,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_classes=Config.NUM_CLASSES,
        )

        # Stage 2
        self.stage2 = SingleStageTCN(
            num_classes=Config.NUM_CLASSES,
            num_layers=len(Config.SAWTOOTH_DILATIONS),
            hidden_channels=Config.REFINE_CHANNELS,
            dropout=Config.DROPOUT,
        )

        # Stage 3
        self.stage3 = SingleStageTCN(
            num_classes=Config.NUM_CLASSES,
            num_layers=len(Config.SAWTOOTH_DILATIONS),
            hidden_channels=Config.REFINE_CHANNELS,
            dropout=Config.DROPOUT,
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, InputDim)
        Returns:
            (logits_1, logits_2, logits_3)
        """
        # Stage 1
        logits_1 = self.stage1(x)
        probs_1 = F.softmax(logits_1, dim=2)

        # Stage 2 (Input is probabilities from Stage 1)
        logits_2 = self.stage2(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # Stage 3 (Input is probabilities from Stage 2)
        logits_3 = self.stage3(probs_2)

        return logits_1, logits_2, logits_3
