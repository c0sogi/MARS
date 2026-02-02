import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledDenseBlock(nn.Module):
    """
    A single dense block with decoupled spatial and channel mixing.
    Structure:
        Conv1d(k=3) -> LayerNorm -> SiLU -> Conv1d(k=1) -> LayerNorm -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=Config.DROPOUT):
        super().__init__()
        self.spatial_mix = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
        )
        self.channel_mix = nn.Sequential(
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (N, C, L)

        # Spatial Mixing
        out = self.spatial_mix[0](x)  # Conv1d
        out = out.permute(0, 2, 1)  # (N, L, C) for LN
        out = self.spatial_mix[1](out)  # LN
        out = out.permute(0, 2, 1)  # (N, C, L)
        out = self.spatial_mix[2](out)  # SiLU

        # Channel Mixing
        out = self.channel_mix[0](out)  # Conv1d
        out = out.permute(0, 2, 1)  # (N, L, C) for LN
        out = self.channel_mix[1](out)  # LN
        out = out.permute(0, 2, 1)  # (N, C, L)
        out = self.channel_mix[2](out)  # SiLU
        out = self.channel_mix[3](out)  # Dropout

        return out


class DilatedDenseBackbone(nn.Module):
    """
    A backbone consisting of stacked DecoupledDenseBlocks with dense connections.
    Features from all previous blocks are concatenated before feeding into the next block.
    The final output is projected to the latent dimension.
    """

    def __init__(
        self,
        in_channels,
        growth_rate=Config.BACKBONE_GROWTH,
        dilations=Config.BACKBONE_DILATIONS,
        latent_dim=Config.LATENT_DIM,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        curr_dim = in_channels

        for d in dilations:
            blk = DecoupledDenseBlock(curr_dim, growth_rate, d)
            self.blocks.append(blk)
            curr_dim += growth_rate

        self.latent_proj = nn.Conv1d(curr_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (N, C_in, L) - Typically output from a stem

        features = [x]

        for blk in self.blocks:
            # Dense connection: Concatenate all previous features
            inp = torch.cat(features, dim=1)
            new_feat = blk(inp)
            features.append(new_feat)

        # Concatenate everything for final projection
        all_feats = torch.cat(features, dim=1)

        # Project to Latent Dim
        z = self.latent_proj(all_feats)  # (N, Latent, L)

        # Permute to (N, L, Latent) for Interaction Layer
        return z.permute(0, 2, 1)


class FeedbackModule(nn.Module):
    """
    Processes recycled predictions (targets) into feedback embeddings.
    Uses a lightweight TCN structure.
    """

    def __init__(self, in_channels=Config.FEEDBACK_CHANNELS, growth_rate=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
        )
        self.block = nn.Sequential(
            nn.Conv1d(growth_rate, growth_rate, kernel_size=3, padding=2, dilation=2),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (N, L, 5) -> Permute to (N, 5, L)
        x = x.permute(0, 2, 1)

        # Stem
        x = self.stem[0](x)
        x = x.permute(0, 2, 1)
        x = self.stem[1](x)
        x = x.permute(0, 2, 1)
        x = self.stem[2](x)

        # Block
        x = self.block[0](x)
        x = x.permute(0, 2, 1)
        x = self.block[1](x)
        x = x.permute(0, 2, 1)
        x = self.block[2](x)

        # Output: (N, 16, L) -> Permute to (N, L, 16)
        return x.permute(0, 2, 1)


class InteractionLayer(nn.Module):
    """
    Fuses backbone features (Z) with feedback features (E_fb).
    Gathers partner features based on secondary structure.
    Aggregates global context using a Bidirectional GRU.
    """

    def __init__(
        self,
        latent_dim=Config.LATENT_DIM,
        feedback_dim=16,
        rnn_hidden=Config.RNN_HIDDEN,
    ):
        super().__init__()
        # Input to RNN: (Latent + Feedback) for Self + (Latent + Feedback) for Partner
        rnn_input_dim = (latent_dim + feedback_dim) * 2

        self.gru = nn.GRU(
            rnn_input_dim, rnn_hidden, batch_first=True, bidirectional=True
        )

    def forward(self, z, e_fb, partner_indices):
        # z: (N, L, Latent)
        # e_fb: (N, L, Feedback)
        # partner_indices: (N, L)

        # 1. Combine Self Features
        h_self = torch.cat([z, e_fb], dim=-1)  # (N, L, Latent+Feedback)

        # 2. Gather Partner Features
        batch_size, seq_len, _ = h_self.shape

        # Create batch indices for gathering
        batch_idx = (
            torch.arange(batch_size, device=z.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Handle -1 in partner_indices (unpaired bases)
        # Map -1 to 0 temporarily to avoid index errors, then mask result
        p_idx_safe = partner_indices.clone()
        p_idx_safe[p_idx_safe == -1] = 0

        # Gather
        h_partner = h_self[batch_idx, p_idx_safe]  # (N, L, Latent+Feedback)

        # Apply Zero-Mask to unpaired bases
        mask_pair = (partner_indices != -1).unsqueeze(-1).float()
        h_partner = h_partner * mask_pair

        # 3. Fuse Self and Partner
        rnn_in = torch.cat([h_self, h_partner], dim=-1)  # (N, L, Total_Dim)

        # 4. Global Aggregation (GRU)
        rnn_out, _ = self.gru(rnn_in)

        return rnn_out
