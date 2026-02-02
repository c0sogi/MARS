import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import (
    SKELETON_INPUT_DIM,
    AUDIO_N_MELS,
    HIDDEN_DIM,
    DROPOUT,
    KERNEL_SIZE,
    MODEL_OUTPUT_CLASSES,
)


class InputStem(nn.Module):
    """
    Modality-specific processing stem: Linear -> Conv1d -> ReLU -> Dropout.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        # Padding ensures output length equals input length
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, InputDim)
        x = self.project(x)  # (B, T, Hidden)
        # Permute for Conv1d: (B, Hidden, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Permute back: (B, T, Hidden)
        x = x.permute(0, 2, 1)
        return x


class GCINet(nn.Module):
    """
    Global-Conditioned Input-Injected Network (GCI-Net).
    Features:
    - Independent Input Stems
    - Multi-View Global Anchoring (GAP + GMP)
    - Global-Conditioned Gating
    - Dual-Injected BiGRU Backbone
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Input Stems
        self.skel_stem = InputStem(SKELETON_INPUT_DIM, HIDDEN_DIM, KERNEL_SIZE, DROPOUT)
        self.audio_stem = InputStem(AUDIO_N_MELS, HIDDEN_DIM, KERNEL_SIZE, DROPOUT)

        # Fused dimension (Concatenation of two stems)
        self.fused_dim = 2 * HIDDEN_DIM
        self.ln_fuse = nn.LayerNorm(self.fused_dim)

        # 2. Multi-View Anchor & Gating
        # Anchor = GAP(Fused) || GMP(Fused)
        self.anchor_dim = 2 * self.fused_dim

        # Gating Projections: Gate = Sigmoid(Wx * X + Wg * G)
        self.gate_x_proj = nn.Linear(self.fused_dim, self.fused_dim)
        self.gate_g_proj = nn.Linear(self.anchor_dim, self.fused_dim)

        # 3. Dual-Injected Backbone
        # Layer 1: Standard BiGRU
        self.gru1 = nn.GRU(
            self.fused_dim, HIDDEN_DIM, batch_first=True, bidirectional=True
        )
        # Output of GRU1 is (B, T, 2*Hidden)

        # Layer 2 Injection Projections
        # Input to GRU2 = H1 + Proj(Y) + Proj(G)
        # Target dimension is 2*HIDDEN_DIM (matching H1 size)
        self.proj_local_y = nn.Linear(self.fused_dim, 2 * HIDDEN_DIM)
        self.proj_global_g = nn.Linear(self.anchor_dim, 2 * HIDDEN_DIM)

        # Layer 2: BiGRU
        self.gru2 = nn.GRU(
            2 * HIDDEN_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True
        )

        # 4. Output Head
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, MODEL_OUTPUT_CLASSES),
        )

    def forward(self, skeleton, audio, lengths):
        """
        Args:
            skeleton: (B, T, 60)
            audio: (B, T, 64)
            lengths: (B,)
        Returns:
            logits: (B, T, NumClasses)
        """
        B, T, _ = skeleton.shape
        device = skeleton.device

        # --- 1. Feature Extraction & Fusion ---
        skel_feat = self.skel_stem(skeleton)  # (B, T, H)
        audio_feat = self.audio_stem(audio)  # (B, T, H)

        fused = torch.cat([skel_feat, audio_feat], dim=2)  # (B, T, 2H)
        fused = self.ln_fuse(fused)

        # --- 2. Multi-View Anchoring ---
        # Create mask for valid time steps: (B, T, 1)
        mask = torch.arange(T, device=device).expand(B, T) < lengths.unsqueeze(1)
        mask_float = mask.unsqueeze(-1).float()

        # Global Average Pooling (GAP)
        sum_feat = torch.sum(fused * mask_float, dim=1)  # (B, 2H)
        sum_len = torch.sum(mask_float, dim=1)  # (B, 1)
        gap = sum_feat / (sum_len + 1e-8)

        # Global Max Pooling (GMP)
        # Fill padding with very small number before max
        fused_masked_for_max = fused.masked_fill(~mask.unsqueeze(-1), -1e9)
        gmp = torch.max(fused_masked_for_max, dim=1)[0]  # (B, 2H)

        # Construct Anchor
        anchor = torch.cat([gap, gmp], dim=1)  # (B, 4H)

        # --- 3. Global-Conditioned Gating ---
        # Gate = Sigmoid(Wx * X + Wg * G)
        gate_x = self.gate_x_proj(fused)  # (B, T, 2H)
        gate_g = self.gate_g_proj(anchor).unsqueeze(1)  # (B, 1, 2H)
        gate = torch.sigmoid(gate_x + gate_g)

        # Apply Gate
        y = fused * gate  # (B, T, 2H)

        # --- 4. Backbone Layer 1 ---
        # Pack sequence for efficient RNN processing
        packed_input = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h1, _ = self.gru1(packed_input)
        h1, _ = pad_packed_sequence(
            packed_h1, batch_first=True, total_length=T
        )  # (B, T, 2H)

        # --- 5. Backbone Layer 2 (Dual Injection) ---
        # Input2 = H1 + Proj(Y) + Proj(G)
        proj_y = self.proj_local_y(y)  # (B, T, 2H)
        proj_g = self.proj_global_g(anchor).unsqueeze(1)  # (B, 1, 2H)

        input2 = h1 + proj_y + proj_g

        packed_input2 = pack_padded_sequence(
            input2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h2, _ = self.gru2(packed_input2)
        h2, _ = pad_packed_sequence(
            packed_h2, batch_first=True, total_length=T
        )  # (B, T, 2H)

        # --- 6. Classification Head ---
        logits = self.head(h2)  # (B, T, Classes)

        return logits
