import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelWiseGatedInteraction(nn.Module):
    """
    Implements the Channel-Wise Gated Structural-Refinement mechanism.
    Equation: h'_i = h_i + sigma(W_gate * [h_i; h_j] + b) * (W_proj * h_j)

    This module allows the network to selectively gate information flow from
    structurally paired bases (j) to the current base (i) on a per-channel basis.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Gate Network: Takes concatenated [h_i; h_j] -> Gate vector z_ij
        self.gate_net = nn.Linear(dim * 2, dim)
        # Projection Network: Transforms h_j before injection
        self.proj_net = nn.Linear(dim, dim)

    def forward(self, h, adjacency):
        """
        Args:
            h: Hidden states tensor of shape (Batch, Seq_Len, Dim)
            adjacency: Adjacency indices of shape (Batch, Seq_Len).
                       Values are indices of paired bases. -1 indicates unpaired.
        Returns:
            h_prime: Refined hidden states of shape (Batch, Seq_Len, Dim)
        """
        batch_size, seq_len, _ = h.shape

        # 1. Prepare Adjacency for Gathering
        # Create a boolean mask for valid pairs (where adjacency != -1)
        valid_mask = adjacency != -1  # Shape: (B, L)

        # Clone adjacency to avoid in-place modification issues
        # Replace -1 with 0 to prevent index out-of-bounds during gather.
        # These dummy values will be masked out later.
        safe_indices = adjacency.clone()
        safe_indices[~valid_mask] = 0

        # 2. Gather Neighbor States (h_j)
        # We want to retrieve h[b, safe_indices[b, i], :] for each i.
        # Expand indices to match the feature dimension for torch.gather
        gather_indices = safe_indices.unsqueeze(-1).expand(
            -1, -1, self.dim
        )  # (B, L, D)
        h_j = torch.gather(h, 1, gather_indices)  # (B, L, D)

        # 3. Compute Channel-Wise Gate (z_ij)
        # Concatenate current state h_i and neighbor state h_j
        concat_h = torch.cat([h, h_j], dim=-1)  # (B, L, 2*D)
        z_ij = torch.sigmoid(self.gate_net(concat_h))  # (B, L, D)

        # 4. Compute Projected Neighbor Info
        h_j_proj = self.proj_net(h_j)  # (B, L, D)

        # 5. Compute Update Vector
        update = z_ij * h_j_proj

        # 6. Apply Masking
        # Zero out the update vector for unpaired positions (where valid_mask is False)
        mask_expanded = valid_mask.unsqueeze(-1).type_as(update)  # (B, L, 1)
        update = update * mask_expanded

        # 7. Residual Injection
        h_prime = h + update

        return h_prime


class CGSRBiGRU(nn.Module):
    """
    Channel-Wise Gated Structural-Refinement BiGRU (CGSR-BiGRU).

    Architecture:
    1. 1D Convolutional Stem (Sequence/Structure/Loop Features -> Embedding)
    2. Deep Recurrent Backbone (3 Blocks)
       - Block 1 & 2: BiGRU -> Channel-Wise Interaction -> LayerNorm
       - Block 3: BiGRU -> LayerNorm (No interaction in final block)
    3. Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # ==============================
        # 1. Convolutional Stem
        # ==============================
        # Projects sparse one-hot inputs (14 channels) to dense embedding (256 channels)
        # Aggregates local context via kernel size 3
        self.conv_stem = nn.Conv1d(
            in_channels=Config.input_channels,
            out_channels=Config.conv_filters,
            kernel_size=Config.conv_kernel_size,
            padding=Config.conv_kernel_size // 2,
        )
        self.gelu = nn.GELU()

        # ==============================
        # 2. Backbone
        # ==============================
        self.hidden_dim = Config.hidden_dim  # 384

        # --- Block 1 ---
        # Input: 256 (from Conv) -> Output: 384 (BiGRU)
        self.gru1 = nn.GRU(
            input_size=Config.conv_filters,
            hidden_size=self.hidden_dim // 2,  # 192 per direction -> 384 total
            bidirectional=True,
            batch_first=True,
        )
        self.inter1 = ChannelWiseGatedInteraction(self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        # --- Block 2 ---
        # Input: 384 -> Output: 384
        self.gru2 = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.inter2 = ChannelWiseGatedInteraction(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # --- Block 3 ---
        # Input: 384 -> Output: 384
        # Note: No interaction module in the final block.
        self.gru3 = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.ln3 = nn.LayerNorm(self.hidden_dim)

        # ==============================
        # 3. Output Head
        # ==============================
        self.dropout = nn.Dropout(Config.dropout)
        self.head = nn.Linear(self.hidden_dim, Config.num_targets)

    def forward(self, x, adjacency):
        """
        Args:
            x: Input features tensor (Batch, Seq_Len, Channels=14)
            adjacency: Adjacency indices (Batch, Seq_Len)

        Returns:
            logits: Predicted degradation rates (Batch, Seq_Len, Num_Targets=5)
        """
        # Permute for Conv1d: (N, L, C) -> (N, C, L)
        x = x.transpose(1, 2)

        # Apply Stem
        x = self.conv_stem(x)
        x = self.gelu(x)

        # Permute back for GRU: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)

        # --- Block 1 ---
        out, _ = self.gru1(x)  # (N, L, 384)
        out = self.inter1(out, adjacency)
        out = self.ln1(out)
        out = self.dropout(out)

        # --- Block 2 ---
        out, _ = self.gru2(out)
        out = self.inter2(out, adjacency)
        out = self.ln2(out)
        out = self.dropout(out)

        # --- Block 3 ---
        out, _ = self.gru3(out)
        # No interaction in final block
        out = self.ln3(out)
        out = self.dropout(out)

        # Head
        logits = self.head(out)

        return logits
