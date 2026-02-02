import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D data (N, C, L).
    Recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class GatedTCNBlock(nn.Module):
    """
    Temporal Convolutional Block with Gated Activation and SE Attention.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(GatedTCNBlock, self).__init__()

        # Padding to maintain temporal dimension with dilation
        # (kernel_size - 1) * dilation // 2 for 'same' padding if stride=1
        # However, for causal/standard TCN, we often pad differently.
        # Here we use 'same' padding logic for non-causal refinement (since we have the full window).
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels * 2,  # *2 for Gated Activation (Content + Gate)
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(out_channels)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x

        out = self.conv(x)

        # Gated Activation: split channels
        out_tanh, out_sigmoid = out.chunk(2, dim=1)
        out = torch.tanh(out_tanh) * torch.sigmoid(out_sigmoid)

        out = self.dropout(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Spatial-Kinematic Sequence Encoder.
    Processes raw features using Bi-GRU.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (N, L, Input_Dim)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)

        # out: (N, L, Hidden*2)
        logits = self.fc(out)
        return logits


class RefinementModule(nn.Module):
    """
    Stage 2 & 3: Attentive Gated Refinement Module.
    Takes class probabilities as input and refines them.
    """

    def __init__(self, num_classes, hidden_dim, num_layers=4, dropout=0.3):
        super(RefinementModule, self).__init__()

        layers = []
        # Input projection: Num_Classes -> Hidden_Dim
        layers.append(nn.Conv1d(num_classes, hidden_dim, 1))

        # Stacked Gated TCN Blocks with increasing dilation
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                GatedTCNBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # Output projection: Hidden_Dim -> Num_Classes
        layers.append(nn.Conv1d(hidden_dim, num_classes, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, L, Num_Classes) -> Transpose to (N, C, L) for Conv1d
        x = x.transpose(1, 2)
        out = self.net(x)
        # Transpose back to (N, L, C)
        out = out.transpose(1, 2)
        return out


class RSKARN(nn.Module):
    """
    Robust Spatial-Kinematic Attentive Refinement Network.
    Three-stage cascaded architecture.
    """

    def __init__(self):
        super(RSKARN, self).__init__()

        self.num_classes = Config.NUM_CLASSES

        # Stage 1: Encoder
        self.stage1 = BiGRUEncoder(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            num_classes=self.num_classes,
            dropout=Config.DROPOUT,
        )

        # Stage 2: Refinement
        # Internal hidden size for refinement can be same as encoder or smaller
        self.stage2 = RefinementModule(
            num_classes=self.num_classes,
            hidden_dim=64,  # Lightweight refinement
            num_layers=4,  # Receptive field cover
            dropout=Config.DROPOUT,
        )

        # Stage 3: Refinement
        self.stage3 = RefinementModule(
            num_classes=self.num_classes,
            hidden_dim=64,
            num_layers=4,
            dropout=Config.DROPOUT,
        )

    def forward(self, x):
        """
        Forward pass returning outputs from all stages for cascaded loss.
        x: (N, L, Input_Dim)
        """
        # Stage 1
        logits_1 = self.stage1(x)
        probs_1 = F.softmax(logits_1, dim=2)

        # Stage 2 (Input is strictly probabilities from Stage 1)
        logits_2 = self.stage2(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # Stage 3 (Input is strictly probabilities from Stage 2)
        logits_3 = self.stage3(probs_2)

        return {
            "logits_1": logits_1,
            "logits_2": logits_2,
            "logits_3": logits_3,
            "probs_3": F.softmax(logits_3, dim=2),  # Final prediction
        }
