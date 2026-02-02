import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D Temporal Sequences.
    Recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (Batch, Channels, Time)
        b, c, t = x.size()
        # Squeeze: Global Average Pooling -> (B, C, 1) -> (B, C)
        y = self.avg_pool(x).view(b, c)
        # Excitation: MLP -> (B, C) -> (B, C, 1)
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y


class GatedTCNBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block with SE Attention.
    Structure: DilatedConv -> Split(Filter, Gate) -> GatedAct -> SE -> 1x1 -> Dropout -> Residual
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedTCNBlock, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )

        self.se = SEBlock(out_channels)
        self.conv_1x1 = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x

        # Dilated Conv
        out = self.conv_dilated(x)

        # Gated Activation
        filter_out, gate_out = out.chunk(2, dim=1)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # Squeeze-and-Excitation
        out = self.se(out)

        # Projection
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return out + residual


class RefinementStage(nn.Module):
    """
    Attentive Gated Refinement Module.
    Takes class probabilities as input and outputs refined class logits.
    """

    def __init__(self, num_classes, hidden_channels, num_layers, kernel_size, dropout):
        super(RefinementStage, self).__init__()

        # Input Projection: NumClasses -> Hidden
        self.conv_in = nn.Conv1d(num_classes, hidden_channels, 1)

        # Stack of Gated TCN Blocks
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                GatedTCNBlock(
                    hidden_channels, hidden_channels, kernel_size, dilation, dropout
                )
            )
        self.layers = nn.Sequential(*layers)

        # Output Projection: Hidden -> NumClasses
        self.conv_out = nn.Conv1d(hidden_channels, num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) -> Transpose to (Batch, NumClasses, Time)
        x = x.permute(0, 2, 1)

        out = self.conv_in(x)
        out = self.layers(out)
        out = self.conv_out(out)

        # Transpose back to (Batch, Time, NumClasses)
        return out.permute(0, 2, 1)


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-Directional GRU.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiGRUEncoder, self).__init__()

        self.projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        b, t, _ = x.size()

        x = self.projection(x)
        x, _ = self.gru(x)
        logits = self.classifier(x)

        return logits


class SA_AKN(nn.Module):
    """
    Structurally-Augmented Attentive Kinematic Network (SA-AKN).
    Three-Stage Cascaded Network:
    1. BiGRU Encoder
    2. Attentive Refinement
    3. Attentive Refinement
    """

    def __init__(self):
        super(SA_AKN, self).__init__()

        # Hyperparameters from Config
        input_dim = Config.INPUT_DIM
        num_classes = Config.NUM_CLASSES

        # Stage 1: BiGRU
        self.stage1 = BiGRUEncoder(
            input_dim=input_dim,
            hidden_dim=Config.GRU_HIDDEN_DIM,
            num_layers=Config.GRU_LAYERS,
            num_classes=num_classes,
        )

        # Stage 2: Refinement
        self.stage2 = RefinementStage(
            num_classes=num_classes,
            hidden_channels=Config.TCN_CHANNELS,
            num_layers=Config.TCN_LAYERS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

        # Stage 3: Refinement (Iterative)
        self.stage3 = RefinementStage(
            num_classes=num_classes,
            hidden_channels=Config.TCN_CHANNELS,
            num_layers=Config.TCN_LAYERS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

    def forward(self, x):
        """
        Forward pass returning logits from all stages for cascaded loss.
        x: (Batch, Time, InputDim)
        """
        # Stage 1
        logits1 = self.stage1(x)
        probs1 = F.softmax(logits1, dim=2)

        # Stage 2 (Input: Probabilities from Stage 1)
        logits2 = self.stage2(probs1)
        probs2 = F.softmax(logits2, dim=2)

        # Stage 3 (Input: Probabilities from Stage 2)
        logits3 = self.stage3(probs2)

        return logits1, logits2, logits3
