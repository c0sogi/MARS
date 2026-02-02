import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from library.config import (
    SKELETON_JOINTS,
    SKELETON_CHANNELS,
    AUDIO_N_MELS,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    CNN_KERNEL_SIZE,
    PYRAMID_LEVELS,
    NUM_CLASSES,
    BATCH_SIZE,
)


class ContextGating(nn.Module):
    """
    Context Gating Mechanism: Y = X * sigmoid(W*X + b)
    Acts as a learnable gate to filter noise and emphasize relevant features.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (Batch, Time, Dim)
        gates = torch.sigmoid(self.fc(x))
        return x * gates


class TemporalPyramidPool(nn.Module):
    """
    Temporal Pyramid Anchoring.
    Extracts fixed-size context vectors from variable-length sequences at multiple scales.
    """

    def __init__(self, input_dim, levels):
        super(TemporalPyramidPool, self).__init__()
        self.levels = levels
        self.input_dim = input_dim
        # Output dimension is input_dim * sum(levels)

    def forward(self, x, lengths):
        # x: (Batch, Time, Dim)
        # lengths: (Batch,)
        batch_size = x.size(0)
        pyramid_features = []

        for i in range(batch_size):
            length = lengths[i]
            if length == 0:
                # Fallback for empty sequence
                feat_dim = self.input_dim * sum(self.levels)
                pyramid_features.append(torch.zeros(feat_dim, device=x.device))
                continue

            # Extract valid sequence and transpose for pooling: (Dim, Time)
            valid_seq = x[i, :length, :].transpose(0, 1)

            level_feats = []
            for level in self.levels:
                # AdaptiveAvgPool1d ensures we get exactly 'level' number of bins
                # Output: (Dim, Level)
                out = F.adaptive_avg_pool1d(valid_seq, output_size=level)
                # Flatten to (Dim * Level)
                level_feats.append(out.view(-1))

            # Concat all levels: (Dim * Sum(Levels))
            concat_feats = torch.cat(level_feats, dim=0)
            pyramid_features.append(concat_feats)

        return torch.stack(pyramid_features, dim=0)


class InputStem(nn.Module):
    """
    Decoupled Input Stem for independent modality processing.
    Linear -> Conv1d -> ReLU -> Dropout
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super(InputStem, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        # Padding to maintain temporal length: k=7 -> p=3
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        x = self.linear(x)

        # Conv1d expects (Batch, Channels, Time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x = self.relu(x)
        x = self.dropout(x)
        return x


class PCA_IIN(nn.Module):
    """
    Pyramid-Context Anchored Input-Injected Network.
    """

    def __init__(self):
        super(PCA_IIN, self).__init__()

        # 1. Decoupled Input Stems
        self.skel_stem = InputStem(
            input_dim=SKELETON_JOINTS * SKELETON_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            kernel_size=CNN_KERNEL_SIZE,
            dropout=DROPOUT,
        )

        self.audio_stem = InputStem(
            input_dim=AUDIO_N_MELS,
            hidden_dim=HIDDEN_DIM,
            kernel_size=CNN_KERNEL_SIZE,
            dropout=DROPOUT,
        )

        # 2. Gated Fusion
        # Concatenation of two stems -> 2 * HIDDEN_DIM
        self.fusion_linear = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
        self.layer_norm = nn.LayerNorm(HIDDEN_DIM)
        self.context_gating = ContextGating(HIDDEN_DIM)

        # 3. Temporal Pyramid Anchor
        self.pyramid_pool = TemporalPyramidPool(HIDDEN_DIM, PYRAMID_LEVELS)
        pyramid_out_dim = HIDDEN_DIM * sum(PYRAMID_LEVELS)

        # 4. Pyramid-Injected Recurrent Backbone
        # BiGRU Layer 1
        self.gru1 = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=HIDDEN_DIM,
            bidirectional=True,
            batch_first=True,
        )

        # Projections for Layer 2 Injection
        # GRU1 output is 2 * HIDDEN_DIM (Bidirectional)
        gru_out_dim = 2 * HIDDEN_DIM

        self.proj_local = nn.Linear(HIDDEN_DIM, gru_out_dim)
        self.proj_pyramid = nn.Linear(pyramid_out_dim, gru_out_dim)

        # BiGRU Layer 2
        # Input is sum of (GRU1_out, Proj_Local, Proj_Pyramid) -> dim is gru_out_dim
        self.gru2 = nn.GRU(
            input_size=gru_out_dim,
            hidden_size=HIDDEN_DIM,
            bidirectional=True,
            batch_first=True,
        )

        # 5. Output Head
        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, skel, audio, lengths):
        # skel: (B, T, 60)
        # audio: (B, T, 64)
        # lengths: (B,) tensor

        # --- 1. Stems ---
        skel_feat = self.skel_stem(skel)
        audio_feat = self.audio_stem(audio)

        # --- 2. Fusion ---
        concat = torch.cat([skel_feat, audio_feat], dim=2)
        fused = self.fusion_linear(concat)
        fused = self.layer_norm(fused)
        fused = self.context_gating(fused)  # Y

        # --- 3. Pyramid Anchor ---
        # P: (B, PyramidDim)
        pyramid_anchor = self.pyramid_pool(fused, lengths)

        # --- 4. Backbone ---

        # Ensure lengths are on CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu().to(torch.int64)

        # Pack sequence for GRU1
        packed_input = pack_padded_sequence(
            fused, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # GRU 1
        packed_h1, _ = self.gru1(packed_input)

        # Unpack to add injections
        h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        # h1: (B, T, 2*Hidden)

        # Prepare injections
        # Local injection: Project Y to match H1 dim
        inj_local = self.proj_local(fused)

        # Pyramid injection: Project P to match H1 dim and broadcast
        # P is (B, PyramidDim) -> (B, 1, 2*Hidden)
        inj_pyramid = self.proj_pyramid(pyramid_anchor).unsqueeze(1)

        # Sum components (Dual Injection)
        # Note: Broadcasting handles the time dimension for inj_pyramid
        input_2 = h1 + inj_local + inj_pyramid

        # Pack for GRU 2 (masking handles the padding values in input_2)
        packed_input_2 = pack_padded_sequence(
            input_2, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_h2, _ = self.gru2(packed_input_2)

        # Unpack final output
        h2, _ = pad_packed_sequence(packed_h2, batch_first=True)
        # h2: (B, T, 2*Hidden)

        # --- 5. Head ---
        logits = self.head(h2)  # (B, T, NumClasses)

        return logits
