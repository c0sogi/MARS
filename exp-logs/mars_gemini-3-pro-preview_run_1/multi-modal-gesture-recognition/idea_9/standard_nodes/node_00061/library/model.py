import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library import config


class TemporalStream(nn.Module):
    """
    Independent temporal processing stream for a single modality.
    Structure: Linear -> Conv1d -> ReLU -> Dropout.
    Cite solution_lesson_node_00059: Lightweight independent feature extractors.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(TemporalStream, self).__init__()

        # Projection to hidden dimension
        self.project = nn.Linear(input_dim, hidden_dim)

        # Temporal Convolution (Mid-Fusion preparation)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, InputDim)
        Returns:
            (Batch, Time, HiddenDim)
        """
        # 1. Projection
        x = self.project(x)  # (B, T, H)

        # 2. Conv1d (Requires B, C, T)
        x = x.permute(0, 2, 1)  # (B, H, T)
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (B, T, H)

        return x


class ContextGating(nn.Module):
    """
    Context Gating mechanism: Y = X * Sigmoid(W*X + b)
    Dynamically re-weights features based on context.
    """

    def __init__(self, input_dim):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sigmoid(self.fc(x))
        return x * gates


class CGRNet(nn.Module):
    """
    Context-Gated Residual Network (CGR-GRU).
    Cite solution_lesson_node_00036: Context Gating and Residual Connections.
    Cite solution_lesson_node_00059: Mid-Fusion with shared backbone.
    """

    def __init__(self):
        super(CGRNet, self).__init__()

        # --- 1. Lightweight Temporal Stems ---
        # Skeleton Stream
        self.skel_stream = TemporalStream(
            input_dim=60,
            hidden_dim=config.HIDDEN_DIM_STREAM,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT_RATE,
        )

        # Audio Stream
        self.audio_stream = TemporalStream(
            input_dim=config.N_MFCC,
            hidden_dim=config.HIDDEN_DIM_STREAM,
            kernel_size=config.KERNEL_SIZE,
            dropout=config.DROPOUT_RATE,
        )

        # Fusion Dimension: 256 + 256 = 512
        self.fusion_dim = config.HIDDEN_DIM_STREAM * 2

        # --- 2. Gated Fusion ---
        self.ln = nn.LayerNorm(self.fusion_dim)
        self.cg = ContextGating(self.fusion_dim)

        # --- 3. Residual BiGRU Backbone ---
        # Layer 1
        self.gru1 = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=config.HIDDEN_DIM_BACKBONE,
            batch_first=True,
            bidirectional=True,
        )

        # Layer 2
        self.gru2 = nn.GRU(
            input_size=config.HIDDEN_DIM_BACKBONE * 2,  # 512
            hidden_size=config.HIDDEN_DIM_BACKBONE,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # --- 4. Non-Linear Classification Head ---
        self.head = nn.Sequential(
            nn.Linear(2 * config.HIDDEN_DIM_BACKBONE, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        # 1. Independent Feature Extraction (Mid-Fusion Prep)
        skel_feat = self.skel_stream(skeleton)  # (B, T, 256)
        audio_feat = self.audio_stream(audio)  # (B, T, 256)

        # 2. Fusion & Gating
        fused = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, 512)
        fused = self.ln(fused)
        fused = self.cg(fused)

        # 3. Residual Recurrent Backbone
        packed = pack_padded_sequence(
            fused, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Layer 1
        packed_out1, _ = self.gru1(packed)
        out1, _ = pad_packed_sequence(
            packed_out1, batch_first=True, total_length=fused.size(1)
        )

        # Residual 1 (Input 512 + Output 512)
        res1 = fused + out1
        res1 = self.dropout(res1)

        # Layer 2
        packed_res1 = pack_padded_sequence(
            res1, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out2, _ = self.gru2(packed_res1)
        out2, _ = pad_packed_sequence(
            packed_out2, batch_first=True, total_length=fused.size(1)
        )

        # Residual 2
        res2 = res1 + out2

        # 4. Classification
        logits = self.head(res2)

        return logits
