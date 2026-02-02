import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    HIDDEN_DIM,
    GROWTH_RATE,
    FEEDBACK_GROWTH_RATE,
    DROPOUT,
    NUM_TARGETS,
)


class SpatialStem(nn.Module):
    """
    Processes inputs with a spatial kernel to extract immediate context.
    Structure: Conv1d(k=3) -> LayerNorm -> SiLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same")
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv(x)
        # LayerNorm expects [B, L, C], so we transpose
        out = out.transpose(1, 2)
        out = self.ln(out)
        out = out.transpose(1, 2)
        out = self.act(out)
        return out


class PostActDenseBlock(nn.Module):
    """
    A dense block with post-activation structure and decoupled spatial/channel mixing.
    Structure:
      Dilated Conv(k=3) -> LN -> SiLU -> Pointwise Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding="same", dilation=dilation
        )
        self.ln1 = nn.LayerNorm(growth_rate)

        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.ln2 = nn.LayerNorm(growth_rate)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, L]

        # 1. Spatial Aggregation (Dilated Conv)
        out = self.conv1(x)
        out = self.ln1(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)

        # 2. Channel Mixing (Pointwise Conv)
        out = self.conv2(out)
        out = self.ln2(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)

        out = self.dropout(out)

        # Dense Connection
        return torch.cat([x, out], dim=1)


class DenseTCN(nn.Module):
    """
    A stack of PostActDenseBlocks with exponentially increasing dilation rates.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        curr_dim = in_channels

        for d in dilations:
            blk = PostActDenseBlock(curr_dim, growth_rate, dilation=d, dropout=dropout)
            self.blocks.append(blk)
            curr_dim += growth_rate

        self.projection = nn.Conv1d(curr_dim, out_channels, kernel_size=1)

    def forward(self, x):
        feat = x
        for blk in self.blocks:
            feat = blk(feat)
        return self.projection(feat)


class DSRDN(nn.Module):
    """
    Dual-Stem Recurrent Dense Network.
    Integrates static sequence features and dynamic feedback predictions.
    """

    def __init__(self, in_channels=18):
        super().__init__()

        # --- Static Branch ---
        # Input: 18 channels (Seq, Struct, Loop, PartnerID)
        self.static_stem = SpatialStem(in_channels, HIDDEN_DIM)

        # Static Backbone: Dilations 1, 2, 4, 8, 16, 32
        self.static_backbone = DenseTCN(
            in_channels=HIDDEN_DIM,
            growth_rate=GROWTH_RATE,
            dilations=[1, 2, 4, 8, 16, 32],
            dropout=DROPOUT,
            out_channels=HIDDEN_DIM,  # Latent Z dim = 64
        )

        # --- Feedback Branch ---
        # Input: 5 channels (Targets)
        self.feedback_stem = SpatialStem(NUM_TARGETS, 32)

        # Feedback Backbone: Dilations 1, 2, 4, 8
        self.feedback_backbone = DenseTCN(
            in_channels=32,
            growth_rate=FEEDBACK_GROWTH_RATE,
            dilations=[1, 2, 4, 8],
            dropout=DROPOUT,
            out_channels=32,  # Feedback Embedding dim = 32
        )

        # --- Interaction & Aggregation ---
        # Input to RNN: (Static(64) + Feedback(32)) * 2 (Self + Partner) = 192
        rnn_input_dim = (HIDDEN_DIM + 32) * 2
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Head: Bidirectional RNN outputs 2 * HIDDEN_DIM
        self.head = nn.Linear(HIDDEN_DIM * 2, NUM_TARGETS)

    def forward_static(self, x):
        """
        Computes static latent representation Z.
        """
        # x: [B, 18, L]
        stem_out = self.static_stem(x)
        z = self.static_backbone(stem_out)  # [B, 64, L]
        return z

    def forward_feedback(self, y_prev):
        """
        Computes feedback embedding E_fb from previous predictions.
        """
        # y_prev: [B, 5, L]

        # Mask unscored channels: deg_pH10 (idx 2) and deg_50C (idx 4)
        # Keep: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Mask vector: [1, 1, 0, 1, 0]
        mask = torch.tensor(
            [1, 1, 0, 1, 0], device=y_prev.device, dtype=y_prev.dtype
        ).view(1, 5, 1)
        y_masked = y_prev * mask

        stem_out = self.feedback_stem(y_masked)
        e_fb = self.feedback_backbone(stem_out)  # [B, 32, L]
        return e_fb

    def forward_interaction(self, z, e_fb, pairs):
        """
        Performs Augmented Gather and RNN aggregation.
        """
        # z: [B, 64, L], e_fb: [B, 32, L]
        # pairs: [B, L] (indices of partners, -1 if unpaired)

        # 1. Concatenate Self Features
        self_feat = torch.cat([z, e_fb], dim=1)  # [B, 96, L]

        # 2. Augmented Gather (Partner Features)
        B, C, L = self_feat.shape

        # Prepare indices for gathering
        # We want to gather from dim=2 (Length)
        # Create batch indices: [[0,0...], [1,1...], ...]
        batch_idx = torch.arange(B, device=z.device).view(B, 1).expand(B, L)

        # Handle -1 in pairs by replacing with 0 temporarily and then masking
        valid_mask = (pairs != -1).unsqueeze(1)  # [B, 1, L]
        safe_pairs = pairs.clone()
        safe_pairs[pairs == -1] = 0

        # Gather partner features
        # self_feat is [B, C, L]. We need self_feat[b, :, pair_idx]
        # Transpose to [B, L, C] for easier gathering or use advanced indexing
        # Using advanced indexing on [B, C, L]:
        # We need to select specific L indices for each batch.
        # self_feat[batch_idx, :, safe_pairs] -> This might flatten C if not careful.
        # Correct approach:
        partner_feat = self_feat[batch_idx, :, safe_pairs]  # Result: [B, L, C]
        partner_feat = partner_feat.transpose(1, 2)  # Back to [B, C, L]

        # Apply mask to zero out features for unpaired bases
        partner_feat = partner_feat * valid_mask

        # 3. Fusion
        combined = torch.cat([self_feat, partner_feat], dim=1)  # [B, 192, L]

        # 4. Global Aggregation (RNN)
        # RNN expects [B, L, Input_Size]
        combined_t = combined.transpose(1, 2)
        rnn_out, _ = self.rnn(combined_t)  # [B, L, 128]

        # 5. Prediction Head
        preds = self.head(rnn_out)  # [B, L, 5]

        return preds.transpose(1, 2)  # [B, 5, L]

    def forward(self, x, pairs, y_prev=None):
        """
        Main forward pass.
        If y_prev is None, initializes with zeros (Pass 1).
        """
        if y_prev is None:
            y_prev = torch.zeros((x.shape[0], NUM_TARGETS, x.shape[2]), device=x.device)

        z = self.forward_static(x)
        e_fb = self.forward_feedback(y_prev)
        preds = self.forward_interaction(z, e_fb, pairs)

        return preds, z
