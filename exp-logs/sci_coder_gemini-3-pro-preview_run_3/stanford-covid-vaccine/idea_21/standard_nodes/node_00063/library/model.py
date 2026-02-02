import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelGatedInteraction(nn.Module):
    """
    Non-Linear Channel-Gated Structural Interaction Module.
    Gathers hidden states from paired bases, applies non-linear projection,
    and selectively injects information via a learned gate.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Value Projection (Non-Linear): h_j -> v_ij
        # Projects the neighbor's state into a message vector
        self.val_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(Config.DROPOUT)
        )

        # Gating Mechanism: [h_i; h_j] -> z_ij
        # Determines how much of the message to accept per channel
        self.gate_mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj):
        """
        Args:
            h: (B, L, D) Hidden states from the sequence model
            adj: (B, L) Adjacency indices. -1 indicates unpaired bases.
        """
        B, L, D = h.shape

        # 1. Gather Neighbor States (h_j)
        # Handle -1 by clamping to 0 and masking later to zero out unpaired contributions
        mask = (adj != -1).unsqueeze(-1).float()  # (B, L, 1)
        adj_clamped = adj.clone()
        adj_clamped[adj == -1] = 0

        # Create batch indices for gathering: (B, L)
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, L)
        h_j = h[batch_idx, adj_clamped]  # (B, L, D)

        # 2. Non-Linear Value Projection
        v_ij = self.val_proj(h_j)

        # 3. Channel-Wise Gating
        # Concatenate self (h_i) and neighbor (h_j) context
        cat = torch.cat([h, h_j], dim=-1)  # (B, L, 2D)

        # Non-linear mixing followed by projection to sigmoid gate
        gate_hidden = F.gelu(self.gate_mix(cat))
        z_ij = torch.sigmoid(self.gate_proj(gate_hidden))  # (B, L, D)

        # 4. Selective Injection
        # Apply gate to value, ensure unpaired bases receive 0 injection
        injection = z_ij * v_ij * mask

        # 5. Residual Connection + Layer Normalization
        return self.norm(h + injection)


class BiGRU_Block(nn.Module):
    """
    A single block of the backbone containing a Bidirectional GRU
    and an optional Structural Interaction Module.
    """

    def __init__(self, hidden_dim, use_interaction=True):
        super().__init__()
        # BiGRU: We set hidden_size = hidden_dim // 2 so that the
        # concatenated bidirectional output has size hidden_dim.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.use_interaction = use_interaction
        if use_interaction:
            self.interaction = ChannelGatedInteraction(hidden_dim)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x, adj):
        # x: (B, L, D)
        out, _ = self.gru(x)  # (B, L, D)
        out = self.dropout(out)

        if self.use_interaction:
            out = self.interaction(out, adj)

        return out


class RNAModel(nn.Module):
    """
    Deep Iterative Structural-Refinement Model.
    Consists of a Convolutional Stem, a High-Capacity BiGRU Backbone
    with Interleaved Channel-Gated Structural Injection, and a Linear Head.
    """

    def __init__(self):
        super().__init__()

        # Convolutional Stem
        # Projects sparse one-hot inputs (14 channels) to dense embedding space
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_DIM,
                out_channels=Config.CNN_FILTERS,
                kernel_size=Config.CNN_KERNEL,
                padding=Config.CNN_KERNEL // 2,
            ),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # Projection to match Backbone hidden dimension
        self.proj = nn.Linear(Config.CNN_FILTERS, Config.HIDDEN_DIM)

        # High-Capacity Backbone (3 Blocks)
        # Interaction is applied in the first 2 blocks to refine structure,
        # the last block is a standard BiGRU for final sequence integration.
        self.blocks = nn.ModuleList(
            [
                BiGRU_Block(Config.HIDDEN_DIM, use_interaction=True),
                BiGRU_Block(Config.HIDDEN_DIM, use_interaction=True),
                BiGRU_Block(Config.HIDDEN_DIM, use_interaction=False),
            ]
        )

        # Output Head
        # Predicts 5 target values per position
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.OUTPUT_DIM)

    def forward(self, x, adj):
        # x: (B, L, 14) - One-hot encoded features
        # adj: (B, L) - Structural adjacency list

        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Project to hidden dimension
        x = self.proj(x)

        # Pass through backbone blocks
        for block in self.blocks:
            x = block(x, adj)

        # Final prediction head
        out = self.head(x)
        return out
