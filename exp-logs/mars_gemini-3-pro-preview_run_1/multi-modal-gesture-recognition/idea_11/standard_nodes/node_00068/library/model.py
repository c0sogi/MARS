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


class ResidualBiGRU(nn.Module):
    """
    Projected Residual BiGRU Block.
    Structure: Residual(Projection(x)) + LayerNorm(Dropout(BiGRU(x)))
    Cite solution_lesson_node_00061: Decoupling capacity via projected residuals.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ResidualBiGRU, self).__init__()

        # BiGRU: Output dimension is 2 * hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)

        output_dim = 2 * hidden_dim
        self.ln = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual connection: Project input if dimensions mismatch
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        res = self.residual_proj(x)

        out, _ = self.gru(x)
        out = self.dropout(out)
        out = self.ln(out)

        return out + res


class CGRNet(nn.Module):
    """
    Context-Gated Residual Network.
    Cite solution_lesson_node_00067: Gate-Once strategy.
    Cite solution_lesson_node_00036: Context-Gated Residual GRU.
    """

    def __init__(self):
        super(CGRNet, self).__init__()

        # 1. Decoupled Input Stems
        self.skel_stem = InputStem(
            input_dim=Config.SKELETON_INPUT_DIM,
            output_dim=64,
            kernel_size=Config.KERNEL_SIZE_SKELETON,
            dropout=Config.DROPOUT,
        )

        self.audio_stem = InputStem(
            input_dim=Config.N_MFCC,
            output_dim=32,
            kernel_size=Config.KERNEL_SIZE_AUDIO,
            dropout=Config.DROPOUT,
        )

        fusion_dim = 64 + 32  # 96

        # 2. Gated Fusion (Gate Once)
        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fusion_gating = ContextGating(fusion_dim)

        # 3. Residual Backbone
        self.backbone = nn.ModuleList()
        current_dim = fusion_dim
        hidden_dim = Config.HIDDEN_DIM  # e.g., 256

        for _ in range(Config.NUM_LAYERS):
            block = ResidualBiGRU(current_dim, hidden_dim, Config.DROPOUT)
            self.backbone.append(block)
            current_dim = 2 * hidden_dim  # Output of BiGRU is 2*Hidden

        # 4. Output Head
        self.head = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(current_dim // 2, Config.NUM_CLASSES),
        )

    def forward(self, skel, audio):
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
