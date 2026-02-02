import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ContextGating(nn.Module):
    """
    Context Gating Mechanism: Y = X * Sigmoid(W * X + b)
    Dynamically re-weights features based on their context to suppress noise.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.gate = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (Batch, Time, Dim)
        gate = torch.sigmoid(self.gate(x))
        return x * gate


class InputStem(nn.Module):
    """
    Modality-specific processing stem.
    Structure: Linear -> Conv1d -> ReLU -> Dropout
    """

    def __init__(self, input_dim, output_dim, kernel_size, dropout=0.1):
        super(InputStem, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Padding ensures the temporal dimension remains consistent (L_out = L_in)
        self.conv = nn.Conv1d(
            output_dim, output_dim, kernel_size, padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        x = self.linear(x)

        # Permute for Conv1d: (Batch, Dim, Time)
        x = x.transpose(1, 2)

        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Permute back: (Batch, Time, Dim)
        x = x.transpose(1, 2)
        return x


class GatedProjectedResidualBlock(nn.Module):
    """
    Recursive Gated-Residual Block.
    Structure: BiGRU -> Projection -> LN -> ContextGating -> Residual Add
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GatedProjectedResidualBlock, self).__init__()

        # BiGRU: Output dimension is hidden_dim (hidden_dim//2 * 2 directions)
        self.gru = nn.GRU(
            input_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )

        # Projection Layer: Maps BiGRU output to residual dimension and increases capacity
        self.projection = nn.Linear(hidden_dim, hidden_dim)

        self.ln = nn.LayerNorm(hidden_dim)
        self.gating = ContextGating(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual connection: Project input if dimensions mismatch (e.g., first layer)
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # Residual path
        res = self.residual_proj(x)

        # Main path
        out, _ = self.gru(x)  # (B, T, HiddenDim)
        out = self.projection(out)
        out = self.ln(out)
        out = self.gating(out)
        out = self.dropout(out)

        return out + res


class RCGRNet(nn.Module):
    """
    Recursive Context-Gated Residual Network.
    Multi-stream input -> Gated Fusion -> Recursive Backbone -> MLP Head
    """

    def __init__(self):
        super(RCGRNet, self).__init__()

        # 1. Decoupled Input Stems
        # Skeleton Stem: 60 -> 64
        self.skel_stem = InputStem(
            input_dim=Config.SKELETON_INPUT_DIM,
            output_dim=64,
            kernel_size=Config.KERNEL_SIZE_SKELETON,
            dropout=Config.DROPOUT,
        )

        # Audio Stem: 13 -> 32
        self.audio_stem = InputStem(
            input_dim=Config.N_MFCC,
            output_dim=32,
            kernel_size=Config.KERNEL_SIZE_AUDIO,
            dropout=Config.DROPOUT,
        )

        # Fusion Dimensions
        fusion_dim = 64 + 32  # 96

        # 2. Gated Fusion
        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fusion_gating = ContextGating(fusion_dim)

        # 3. Recursive Gated-Residual Backbone
        self.backbone = nn.ModuleList()
        current_dim = fusion_dim
        hidden_dim = Config.HIDDEN_DIM

        for _ in range(Config.NUM_LAYERS):
            self.backbone.append(
                GatedProjectedResidualBlock(current_dim, hidden_dim, Config.DROPOUT)
            )
            current_dim = hidden_dim  # Subsequent layers take hidden_dim

        # 4. Non-Linear Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(hidden_dim // 2, Config.NUM_CLASSES),
        )

    def forward(self, skel, audio):
        """
        Args:
            skel: (Batch, Time, 60)
            audio: (Batch, Time, 13)
        Returns:
            logits: (Batch, Time, NumClasses)
        """
        # Process Stems
        s_feat = self.skel_stem(skel)
        a_feat = self.audio_stem(audio)

        # Concatenate
        x = torch.cat([s_feat, a_feat], dim=2)

        # Fusion Gating
        x = self.fusion_ln(x)
        x = self.fusion_gating(x)

        # Backbone
        for block in self.backbone:
            x = block(x)

        # Output Head
        logits = self.head(x)

        return logits
