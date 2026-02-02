import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    GROWTH_RATE,
    LATENT_DIM,
    DILATIONS,
    DROPOUT,
    FEEDBACK_GROWTH,
    FEEDBACK_OUT_DIM,
    RNN_HIDDEN,
    TARGET_COLS,
    UNSCORED_TARGETS,
)


class HybridInputStem(nn.Module):
    """
    Hybrid Input Stem that combines raw features with spatially convolved features.
    Satisfies Lesson 00125 (Spatial Mixing) and Lesson 00128 (Raw Feature Preservation).
    """

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        # Branch B: Context
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (B, C, L).
        Returns:
            torch.Tensor: Concatenated features of shape (B, 2*C, L).
        """
        # Branch A: Identity (Raw features)
        branch_a = x

        # Branch B: Context
        branch_b = self.conv(x)
        # LayerNorm expects (B, L, C), so we permute
        branch_b = branch_b.permute(0, 2, 1)
        branch_b = self.norm(branch_b)
        branch_b = self.act(branch_b)
        branch_b = branch_b.permute(0, 2, 1)

        # Concatenate along channel dimension
        return torch.cat([branch_a, branch_b], dim=1)


class PostActDenseBlock(nn.Module):
    """
    Post-Activation Dense Block.
    Decouples spatial aggregation (Dilated Conv) from channel mixing (Pointwise Conv).
    Structure: DilatedConv -> LN -> SiLU -> PointwiseConv -> LN -> SiLU -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=DROPOUT):
        super().__init__()
        # Spatial Aggregation
        self.spatial_conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        # Channel Mixing
        self.pointwise_conv = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Spatial Aggregation
        out = self.spatial_conv(x)
        out = out.permute(0, 2, 1)
        out = self.norm1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        # Channel Mixing
        out = self.pointwise_conv(out)
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)

        return self.dropout(out)


class DenseTCN(nn.Module):
    """
    Dense Dilated TCN Backbone.
    Stack of PostActDenseBlocks with dense connections and exponential dilation.
    """

    def __init__(
        self,
        in_channels,
        growth_rate,
        dilations=DILATIONS,
        dropout=DROPOUT,
        out_channels=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for d in dilations:
            block = PostActDenseBlock(current_channels, growth_rate, d, dropout)
            self.blocks.append(block)
            current_channels += growth_rate

        # Optional projection to latent dimension
        self.out_channels = out_channels
        if out_channels is not None:
            self.projection = nn.Conv1d(current_channels, out_channels, 1)
        else:
            self.projection = None

    def forward(self, x):
        features = [x]

        for block in self.blocks:
            # Dense connection: input to block is concatenation of all previous outputs
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Concatenate all features (Input + All Block Outputs)
        total_features = torch.cat(features, dim=1)

        if self.projection is not None:
            return self.projection(total_features)

        return total_features


class FeedbackProcessor(nn.Module):
    """
    Global-Context Feedback Module.
    Processes recycled predictions with channel masking, spatial stem, and lightweight TCN.
    """

    def __init__(
        self,
        input_channels=5,
        growth_rate=FEEDBACK_GROWTH,
        out_channels=FEEDBACK_OUT_DIM,
        dilations=DILATIONS,
        dropout=DROPOUT,
    ):
        super().__init__()

        # Determine indices of unscored targets to mask
        self.mask_indices = [
            i for i, col in enumerate(TARGET_COLS) if col in UNSCORED_TARGETS
        ]

        # Spatial Feedback Stem
        self.stem_conv = nn.Conv1d(
            input_channels, growth_rate, kernel_size=3, padding=1
        )
        self.stem_norm = nn.LayerNorm(growth_rate)
        self.stem_act = nn.SiLU()

        # Feedback Backbone (Lightweight Dense TCN)
        self.backbone = DenseTCN(
            in_channels=growth_rate,
            growth_rate=growth_rate,  # Usually smaller than main backbone
            dilations=dilations,
            dropout=dropout,
            out_channels=out_channels,
        )

    def forward(self, y_prev):
        """
        Args:
            y_prev (torch.Tensor): Recycled predictions of shape (B, 5, L).
        Returns:
            torch.Tensor: Feedback embeddings of shape (B, Out_Dim, L).
        """
        # 1. Channel Masking (Strictly zero out unscored channels)
        masked_y = y_prev.clone()
        if self.mask_indices:
            masked_y[:, self.mask_indices, :] = 0.0

        # 2. Spatial Stem
        out = self.stem_conv(masked_y)
        out = out.permute(0, 2, 1)
        out = self.stem_norm(out)
        out = self.stem_act(out)
        out = out.permute(0, 2, 1)

        # 3. Backbone
        out = self.backbone(out)

        return out


class InteractionModule(nn.Module):
    """
    Interaction & Aggregation Module.
    Combines Self and Partner vectors and aggregates global context via BiGRU.
    """

    def __init__(self, z_dim, fb_dim, rnn_hidden=RNN_HIDDEN):
        super().__init__()
        self.input_dim = (z_dim + fb_dim) * 2  # Self + Partner

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = 2 * rnn_hidden

    def forward(self, z, e_fb, partner_indices):
        """
        Args:
            z (torch.Tensor): Static latent features (B, Z_dim, L).
            e_fb (torch.Tensor): Feedback embeddings (B, FB_dim, L).
            partner_indices (torch.Tensor): Partner index map (B, L). -1 indicates unpaired.
        Returns:
            torch.Tensor: Aggregated sequence features (B, L, 2*RNN_Hidden).
        """
        B, _, L = z.shape

        # 1. Construct Self Vector
        # Concatenate Z and E_fb -> (B, Total_Dim, L)
        h_self = torch.cat([z, e_fb], dim=1)

        # Permute to (B, L, C) for gathering and RNN
        h_self_perm = h_self.permute(0, 2, 1)

        # 2. Construct Partner Vector
        # Create batch indices for advanced indexing: (B, L)
        batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, L)

        # Identify valid partners
        valid_mask = partner_indices != -1

        # Use 0 for invalid indices to prevent gather errors (will be masked later)
        safe_indices = partner_indices.clone()
        safe_indices[~valid_mask] = 0

        # Gather partner features: h_self_perm[b, partner_idx[b, l], :]
        h_partner = h_self_perm[batch_idx, safe_indices]  # (B, L, C)

        # Apply Zero-Mask to unpaired positions
        h_partner = h_partner * valid_mask.unsqueeze(-1).float()

        # 3. Fusion
        # Concatenate Self and Partner vectors
        combined = torch.cat([h_self_perm, h_partner], dim=2)  # (B, L, 2*C)

        # 4. Global Aggregation
        rnn_out, _ = self.rnn(combined)  # (B, L, 2*Hidden)

        return rnn_out
