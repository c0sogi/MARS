import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    SKELETON_INPUT_DIM,
    AUDIO_INPUT_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT_RATE,
    KERNEL_SIZE_STEM,
    NUM_CLASSES,
)


class ContextGating(nn.Module):
    """
    Context Gating Module: Y = X * sigmoid(W X + b)
    Suppresses noise and emphasizes relevant features.
    """

    def __init__(self, dim):
        super(ContextGating, self).__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (Batch, Time, Dim)
        gates = torch.sigmoid(self.gate(x))
        return x * gates


class InputStem(nn.Module):
    """
    Decoupled Input Stem: Linear -> Conv1d -> ReLU -> Dropout
    Processes raw modality inputs into a latent representation.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(InputStem, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        # Padding to maintain temporal dimension: padding = (k - 1) / 2
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        x = self.linear(x)  # (Batch, Time, HiddenDim)

        # Transpose for Conv1d: (Batch, HiddenDim, Time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.relu(x)

        # Transpose back: (Batch, Time, HiddenDim)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x


class GatedResidualBlock(nn.Module):
    """
    Deeply-Gated Residual Block:
    BiGRU -> Linear -> LayerNorm -> Recursive Context Gating -> Residual
    """

    def __init__(self, hidden_dim, dropout):
        super(GatedResidualBlock, self).__init__()
        # BiGRU outputs 2 * (hidden_dim // 2) = hidden_dim
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gating = ContextGating(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, HiddenDim)
        residual = x

        out, _ = self.gru(x)
        out = self.linear(out)
        out = self.norm(out)
        out = self.gating(out)
        out = self.dropout(out)

        return out + residual


class DGR_RN(nn.Module):
    """
    Deeply-Gated Residual Recurrent Network (DGR-RN)
    Multi-Stream Hybrid Network with Recursive Context Gating.
    """

    def __init__(self):
        super(DGR_RN, self).__init__()

        # 1. Decoupled Input Stems
        self.skel_stem = InputStem(
            SKELETON_INPUT_DIM, HIDDEN_DIM, KERNEL_SIZE_STEM, DROPOUT_RATE
        )
        self.audio_stem = InputStem(
            AUDIO_INPUT_DIM, HIDDEN_DIM, KERNEL_SIZE_STEM, DROPOUT_RATE
        )

        # 2. Gated Fusion
        # Concatenation of two stems (Hidden + Hidden)
        fusion_dim = HIDDEN_DIM * 2
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_gating = ContextGating(fusion_dim)

        # Project back to HIDDEN_DIM for the backbone
        self.fusion_projection = nn.Linear(fusion_dim, HIDDEN_DIM)
        self.fusion_dropout = nn.Dropout(DROPOUT_RATE)

        # 3. Deeply-Gated Residual Backbone
        self.backbone = nn.ModuleList(
            [GatedResidualBlock(HIDDEN_DIM, DROPOUT_RATE) for _ in range(NUM_LAYERS)]
        )

        # 4. Non-Linear Output Head
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, skeleton, audio):
        """
        Args:
            skeleton: (Batch, Time, 60)
            audio: (Batch, Time, 13)
        Returns:
            logits: (Batch, Time, NumClasses)
        """
        # Pass through stems
        skel_feat = self.skel_stem(skeleton)
        audio_feat = self.audio_stem(audio)

        # Fusion
        fused = torch.cat([skel_feat, audio_feat], dim=-1)  # (B, T, 2*H)
        fused = self.fusion_norm(fused)
        fused = self.fusion_gating(fused)

        # Project to backbone dimension
        x = self.fusion_projection(fused)
        x = F.relu(x)
        x = self.fusion_dropout(x)

        # Backbone
        for block in self.backbone:
            x = block(x)

        # Output Head
        logits = self.head(x)

        return logits
