import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import (
    INPUT_DIM_POS,
    INPUT_DIM_VEL,
    INPUT_DIM_AUDIO,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
    USE_BIDIRECTIONAL,
)


class KinematicStem(nn.Module):
    """
    Processes a single modality stream (Position, Velocity, or Audio).
    Structure: Linear -> Permute -> Conv1d -> ReLU -> Dropout -> Permute
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(KinematicStem, self).__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        # Kernel size 7, padding 3 preserves temporal length
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (Batch, Time, Input_Dim)
        x = self.project(x)  # (Batch, Time, Hidden_Dim)

        # Permute for Conv1d: (Batch, Hidden_Dim, Time)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Permute back: (Batch, Time, Hidden_Dim)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x


class ContextGating(nn.Module):
    """
    Context Gating mechanism: Y = X * Sigmoid(W*X + b)
    Dynamically re-weights features based on global context.
    """

    def __init__(self, input_dim):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, Dim)
        gates = self.sigmoid(self.fc(x))
        return x * gates


class ResidualBiGRU(nn.Module):
    """
    Bidirectional GRU with a residual connection from input to output.
    Since input dim and output dim might differ, a projection is applied to the residual.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(ResidualBiGRU, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Projection for residual connection if dimensions don't match
        # Output of BiGRU is hidden_dim * 2
        self.output_dim = hidden_dim * 2
        if input_dim != self.output_dim:
            self.residual_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.residual_proj = nn.Identity()

        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x, lengths):
        # x: (Batch, Time, Input_Dim)

        # Pack sequence for RNN efficiency
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed_input)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )

        # Residual Connection
        res = self.residual_proj(x)
        output = output + res
        output = self.norm(output)

        return output


class KAGRN(nn.Module):
    """
    Kinematic-Aware Gated Residual Network (KA-GRN)

    1. Tri-Stream Input (Position, Velocity, Audio)
    2. Gated Fusion
    3. Residual Recurrent Backbone
    4. Multi-Task Heads (Classification + Boundary)
    """

    def __init__(self):
        super(KAGRN, self).__init__()

        # 1. Input Stems
        # We project each stream to HIDDEN_DIM/2 to keep fusion dimension manageable
        stem_dim = HIDDEN_DIM // 2
        self.pos_stem = KinematicStem(INPUT_DIM_POS, stem_dim, DROPOUT)
        self.vel_stem = KinematicStem(INPUT_DIM_VEL, stem_dim, DROPOUT)
        self.audio_stem = KinematicStem(INPUT_DIM_AUDIO, stem_dim, DROPOUT)

        # Fusion Dimension: 3 streams * stem_dim
        self.fusion_dim = stem_dim * 3

        # 2. Gated Fusion
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.context_gating = ContextGating(self.fusion_dim)
        self.dropout = nn.Dropout(DROPOUT)

        # 3. Backbone
        self.rnn = ResidualBiGRU(
            input_dim=self.fusion_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )

        rnn_out_dim = HIDDEN_DIM * 2

        # 4. Heads
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

        # Boundary Head (Auxiliary Task)
        self.boundary_head = nn.Sequential(
            nn.Linear(rnn_out_dim, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, pos, vel, audio, lengths):
        """
        Args:
            pos: (Batch, Time, 60)
            vel: (Batch, Time, 60)
            audio: (Batch, Time, 13)
            lengths: (Batch)
        Returns:
            logits: (Batch, Time, Num_Classes)
            boundaries: (Batch, Time, 1)
        """
        # 1. Process Stems
        p_feat = self.pos_stem(pos)
        v_feat = self.vel_stem(vel)
        a_feat = self.audio_stem(audio)

        # 2. Fusion
        # Concatenate along feature dimension
        fused = torch.cat([p_feat, v_feat, a_feat], dim=2)
        fused = self.fusion_norm(fused)
        fused = self.context_gating(fused)
        fused = self.dropout(fused)

        # 3. Backbone
        rnn_out = self.rnn(fused, lengths)

        # 4. Heads
        class_logits = self.classifier(rnn_out)
        boundary_logits = self.boundary_head(rnn_out)

        return class_logits, boundary_logits
