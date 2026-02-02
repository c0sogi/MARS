import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class ContextGating(nn.Module):
    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        # x: (Batch, Time, Dim)
        gate = torch.sigmoid(self.fc(x))
        return x * gate


class MVAIIN(nn.Module):
    def __init__(self):
        super(MVAIIN, self).__init__()

        # Hyperparameters
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_classes = Config.NUM_CLASSES
        self.dropout_rate = Config.DROPOUT

        # Stem Dimensions (Projecting to half hidden dim each for balanced fusion)
        self.stem_dim = self.hidden_dim // 2

        # ==========================================
        # 1. Decoupled Input Stems
        # ==========================================

        # Skeleton Stem
        # Linear -> Conv1d(k=7) -> ReLU -> Dropout
        self.skel_linear = nn.Linear(Config.SKELETON_INPUT_DIM, self.stem_dim)
        self.skel_conv = nn.Conv1d(
            in_channels=self.stem_dim,
            out_channels=self.stem_dim,
            kernel_size=Config.KERNEL_SIZE_SKELETON,
            padding=Config.KERNEL_SIZE_SKELETON // 2,
        )
        self.skel_dropout = nn.Dropout(self.dropout_rate)

        # Audio Stem
        # Linear -> Conv1d(k=5) -> ReLU -> Dropout
        self.audio_linear = nn.Linear(Config.AUDIO_INPUT_DIM, self.stem_dim)
        self.audio_conv = nn.Conv1d(
            in_channels=self.stem_dim,
            out_channels=self.stem_dim,
            kernel_size=Config.KERNEL_SIZE_AUDIO,
            padding=Config.KERNEL_SIZE_AUDIO // 2,
        )
        self.audio_dropout = nn.Dropout(self.dropout_rate)

        # ==========================================
        # 2. Fusion & Gating
        # ==========================================
        self.fusion_dim = self.stem_dim * 2
        self.fusion_ln = nn.LayerNorm(self.fusion_dim)
        self.context_gating = ContextGating(self.fusion_dim)

        # ==========================================
        # 3. Dual-Injected Backbone
        # ==========================================

        # Layer 1
        self.gru1 = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # Output of GRU1 (Bi) is 2 * hidden_dim
        self.l1_out_dim = self.hidden_dim * 2

        # Projections for Injection
        # Project Local (Y) to match L1 output
        self.proj_local = nn.Linear(self.fusion_dim, self.l1_out_dim)

        # Project Global Anchor (Mean + Max) to match L1 output
        # Anchor dim = fusion_dim * 2 (Mean + Max)
        self.anchor_dim = self.fusion_dim * 2
        self.proj_global = nn.Linear(self.anchor_dim, self.l1_out_dim)

        # Layer 2
        # Input size matches L1 output size because we sum them
        self.gru2 = nn.GRU(
            input_size=self.l1_out_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.l2_out_dim = self.hidden_dim * 2

        # ==========================================
        # 4. Output Head
        # ==========================================
        self.head = nn.Sequential(
            nn.Linear(self.l2_out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, skeleton, audio, lengths):
        """
        Args:
            skeleton: (B, T, 60)
            audio: (B, T, 64)
            lengths: (B,)
        """
        B, T, _ = skeleton.size()
        device = skeleton.device

        # --- 1. Stems ---

        # Skeleton
        s = self.skel_linear(skeleton)  # (B, T, stem_dim)
        s = s.permute(0, 2, 1)  # (B, stem_dim, T)
        s = self.skel_conv(s)  # (B, stem_dim, T)
        s = F.relu(s)
        s = s.permute(0, 2, 1)  # (B, T, stem_dim)
        s = self.skel_dropout(s)

        # Audio
        a = self.audio_linear(audio)  # (B, T, stem_dim)
        a = a.permute(0, 2, 1)  # (B, stem_dim, T)
        a = self.audio_conv(a)  # (B, stem_dim, T)
        a = F.relu(a)
        a = a.permute(0, 2, 1)  # (B, T, stem_dim)
        a = self.audio_dropout(a)

        # --- 2. Fusion ---

        # Concatenate
        y = torch.cat([s, a], dim=2)  # (B, T, fusion_dim)
        y = self.fusion_ln(y)
        y = self.context_gating(y)  # (B, T, fusion_dim)

        # --- 3. Multi-View Anchor ---

        # Create mask for valid timesteps: (B, T, 1)
        # mask is True for valid, False for padding
        mask = torch.arange(T, device=device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()

        # Global Average Pooling (Mean)
        # Sum valid steps / length
        sum_pooled = torch.sum(y * mask, dim=1)  # (B, fusion_dim)
        mean_pooled = sum_pooled / (lengths.unsqueeze(-1).float() + 1e-8)

        # Global Max Pooling (Max)
        # Set padding to -inf
        y_masked_max = y.clone()
        y_masked_max[mask.expand_as(y) == 0] = -1e9
        max_pooled = torch.max(y_masked_max, dim=1)[0]  # (B, fusion_dim)

        # Concatenate Anchor
        g = torch.cat([mean_pooled, max_pooled], dim=1)  # (B, anchor_dim)

        # --- 4. Layer 1 (BiGRU) ---

        # Pack
        packed_y = pack_padded_sequence(
            y, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward GRU1
        packed_h1, _ = self.gru1(packed_y)

        # Unpack
        h1, _ = pad_packed_sequence(
            packed_h1, batch_first=True, total_length=T
        )  # (B, T, l1_out_dim)

        # --- 5. Dual Injection ---

        # Project Local Input
        proj_y = self.proj_local(y)  # (B, T, l1_out_dim)

        # Project Global Anchor and Expand
        proj_g = self.proj_global(g)  # (B, l1_out_dim)
        proj_g = proj_g.unsqueeze(1).expand(-1, T, -1)  # (B, T, l1_out_dim)

        # Composite Input
        input2 = h1 + proj_y + proj_g

        # --- 6. Layer 2 (BiGRU) ---

        # Pack
        packed_input2 = pack_padded_sequence(
            input2, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward GRU2
        packed_h2, _ = self.gru2(packed_input2)

        # Unpack
        h2, _ = pad_packed_sequence(
            packed_h2, batch_first=True, total_length=T
        )  # (B, T, l2_out_dim)

        # --- 7. Output Head ---
        logits = self.head(h2)  # (B, T, num_classes)

        return logits
