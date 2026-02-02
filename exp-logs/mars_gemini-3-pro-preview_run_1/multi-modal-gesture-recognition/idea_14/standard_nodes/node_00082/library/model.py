import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import (
    SKELETON_INPUT_SIZE,
    AUDIO_INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CLASSES,
    DROPOUT,
    DEVICE,
)


class FeatureStem(nn.Module):
    """
    Decoupled Input Stem: Linear -> Temporal Conv1d(k=7) -> ReLU -> Dropout.
    Processes modalities independently.
    """

    def __init__(self, input_dim, output_dim, dropout=DROPOUT):
        super(FeatureStem, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Kernel size 7 for robust local receptive field
        self.conv = nn.Conv1d(output_dim, output_dim, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, input_dim)
        x = self.linear(x)

        # Permute for Conv1d: (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)

        # Permute back: (B, T, C)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x


class ContextGating(nn.Module):
    """
    Context Gating Mechanism: Y = X * sigmoid(W*X + b).
    Acts as a learnable gate/noise filter.
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T, D)
        gate = self.sigmoid(self.fc(x))
        return x * gate


class GCAResNet(nn.Module):
    """
    Global-Context Anchored Residual Network.
    Features:
    - Dual-Stream Input Stems (Skeleton + Audio)
    - Context Gating Fusion
    - Global Anchor (Masked Global Average Pooling)
    - Dual-Injection BiGRU Backbone
    """

    def __init__(self):
        super(GCAResNet, self).__init__()

        # 1. Decoupled Input Stems
        # We project each modality to HIDDEN_SIZE // 2 so concatenation results in HIDDEN_SIZE
        self.stem_dim = HIDDEN_SIZE // 2
        self.skel_stem = FeatureStem(SKELETON_INPUT_SIZE, self.stem_dim)
        self.audio_stem = FeatureStem(AUDIO_INPUT_SIZE, self.stem_dim)

        # 2. Fusion & Gating
        self.fusion_dim = self.stem_dim * 2
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.context_gating = ContextGating(self.fusion_dim)

        # 3. Recurrent Backbone (Dual-Injection)
        # Layer 1: Takes fused input Y
        self.gru1 = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Projections for Injection
        # GRU1 is bidirectional -> output is 2 * HIDDEN_SIZE
        self.gru_out_dim = HIDDEN_SIZE * 2

        # Project Local Features (Y) to match GRU output dim
        self.proj_local = nn.Linear(self.fusion_dim, self.gru_out_dim)

        # Project Global Anchor (G) to match GRU output dim
        self.proj_global = nn.Linear(self.fusion_dim, self.gru_out_dim)

        # Layer 2: Takes (H1 + Proj(Y) + Proj(G))
        # Input size is gru_out_dim because we sum them up
        self.gru2 = nn.GRU(
            input_size=self.gru_out_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        self.classifier = nn.Sequential(
            nn.Linear(self.gru_out_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        """
        Args:
            skeleton: (B, T, 60)
            audio: (B, T, 20)
            lengths: (B,)
        Returns:
            logits: (B, T, NUM_CLASSES)
        """
        # 1. Feature Extraction
        skel_feat = self.skel_stem(skeleton)  # (B, T, stem_dim)
        audio_feat = self.audio_stem(audio)  # (B, T, stem_dim)

        # 2. Fusion
        # Concatenate along feature dimension
        fused = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, fusion_dim)
        fused = self.fusion_norm(fused)
        Y = self.context_gating(fused)  # (B, T, fusion_dim)

        # 3. Global Anchor Extraction
        # Create mask for padding (B, T, 1)
        max_len = Y.size(1)
        mask = torch.arange(max_len, device=Y.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(2)  # (B, T, 1)

        # Masked Global Average Pooling
        # Sum over time, divide by length
        # Avoid division by zero by clamping lengths
        sum_pooled = torch.sum(Y * mask, dim=1)  # (B, fusion_dim)
        G = sum_pooled / lengths.unsqueeze(1).float().clamp(min=1.0)  # (B, fusion_dim)

        # 4. Backbone - Layer 1
        # Pack sequence for correct RNN handling
        packed_Y = pack_padded_sequence(
            Y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_H1, _ = self.gru1(packed_Y)
        H1, _ = pad_packed_sequence(packed_H1, batch_first=True, total_length=max_len)
        # H1: (B, T, 2*HIDDEN_SIZE)

        # 5. Dual-Stream Injection
        # Input_2 = H1 + Proj_local(Y) + Proj_global(G)
        P_local = self.proj_local(Y)  # (B, T, 2*HIDDEN_SIZE)
        P_global = self.proj_global(G).unsqueeze(
            1
        )  # (B, 1, 2*HIDDEN_SIZE) - broadcast over time

        # Apply mask to P_local and P_global implicitly via packing later,
        # but explicitly zeroing ensures cleanliness
        Input_2 = H1 + P_local + P_global

        # 6. Backbone - Layer 2
        packed_Input2 = pack_padded_sequence(
            Input_2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_H2, _ = self.gru2(packed_Input2)
        H2, _ = pad_packed_sequence(packed_H2, batch_first=True, total_length=max_len)
        # H2: (B, T, 2*HIDDEN_SIZE)

        # 7. Classification
        logits = self.classifier(H2)  # (B, T, NUM_CLASSES)

        return logits
