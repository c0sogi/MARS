import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TemporalBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block.
    Structure:
        Input -> Dilated Conv (splits into Filter/Gate) -> Gating -> Dropout -> 1x1 Conv -> Residual
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()

        # Padding is set to dilation to ensure centered (non-causal) convolution
        # for odd kernel_size (k=3). padding = (k-1) * d / 2 = 1 * d = d.
        self.conv_dilated = nn.Conv1d(
            channels, 2 * channels, kernel_size, padding=dilation, dilation=dilation
        )

        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        residual = x

        out = self.conv_dilated(x)

        # Gated Activation Unit: Tanh * Sigmoid
        P, Q = out.chunk(2, dim=1)
        out = torch.tanh(P) * torch.sigmoid(Q)

        out = self.dropout(out)
        out = self.conv_1x1(out)

        return residual + out


class RefinementStage(nn.Module):
    """
    Monotonic Non-Causal Refinement Stage.
    Takes class probabilities as input, refines them via dilated convolutions,
    and outputs refined class logits.
    """

    def __init__(self, num_classes, hidden_channels, kernel_size, dilations, dropout):
        super(RefinementStage, self).__init__()

        # Project from Probabilities (num_classes) to Hidden Dimension
        self.conv_in = nn.Conv1d(num_classes, hidden_channels, 1)

        # Stack of Temporal Blocks
        self.layers = nn.ModuleList(
            [
                TemporalBlock(hidden_channels, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )

        # Project back to Class Logits
        self.conv_out = nn.Conv1d(hidden_channels, num_classes, 1)

    def forward(self, probs):
        # probs shape: (Batch, Time, NumClasses) -> Needs transpose for Conv1d
        x = probs.transpose(1, 2)  # (B, C, T)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Transpose back: (B, C, T) -> (B, T, C)
        return out.transpose(1, 2)


class KinematicEncoder(nn.Module):
    """
    Stage 1: Physically-Aligned Kinematic Encoder.
    Uses a Bi-GRU to extract temporal features from fused Skeleton and Audio data.
    """

    def __init__(self, input_dim, hidden_size, num_classes):
        super(KinematicEncoder, self).__init__()

        # Bi-directional GRU
        # Hidden size is per direction, so output will be hidden_size * 2
        # Added dropout between layers (Cite solution_lesson_node_00091)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0.0,
        )

        # Projection to initial class logits
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)

        # GRU Output: (Batch, Time, HiddenSize * 2)
        out, _ = self.gru(x)

        # Project to logits
        logits = self.fc(out)

        return logits


class PAKRNet(nn.Module):
    """
    Physically-Aligned Kinematic Refinement Network (PAK-RN).
    Three-Stage Cascaded Network:
    1. Kinematic Encoder (Bi-GRU)
    2. Refinement Stage 1 (TCN)
    3. Refinement Stage 2 (TCN)
    """

    def __init__(self):
        super(PAKRNet, self).__init__()

        # Input Dimension Calculation
        # Skeleton: 20 joints * 9 (Pos+Vel+Acc) = 180
        # Audio: 13 MFCCs
        # Total: 193
        self.input_dim = 193

        # Stage 1
        self.encoder = KinematicEncoder(
            input_dim=self.input_dim,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_classes=Config.NUM_CLASSES,
        )

        # Stage 2
        self.refinement_1 = RefinementStage(
            num_classes=Config.NUM_CLASSES,
            hidden_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
            dropout=Config.DROPOUT,
        )

        # Stage 3
        self.refinement_2 = RefinementStage(
            num_classes=Config.NUM_CLASSES,
            hidden_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
            dropout=Config.DROPOUT,
        )

    def forward(self, x):
        """
        Forward pass returning outputs from all stages for Deep Supervision.

        Args:
            x (torch.Tensor): Input features (Batch, Time, 193)

        Returns:
            list: [logits_1, logits_2, logits_3]
        """
        # Stage 1: Encoder
        logits_1 = self.encoder(x)

        # Convert Logits to Probabilities for Stage 2 input
        probs_1 = F.softmax(logits_1, dim=2)

        # Stage 2: Refinement
        logits_2 = self.refinement_1(probs_1)

        # Convert Logits to Probabilities for Stage 3 input
        probs_2 = F.softmax(logits_2, dim=2)

        # Stage 3: Refinement
        logits_3 = self.refinement_2(probs_2)

        return [logits_1, logits_2, logits_3]
