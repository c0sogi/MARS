import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WindowedInteractionModule(nn.Module):
    """
    Windowed Structural Interaction Module.

    This module gathers hidden states from a local window around the paired base
    for each position, aggregates them, and injects the structural context
    via a gated residual connection.

    Args:
        hidden_dim (int): The dimension of the hidden states (BiGRU output width).
        window_size (int): The size of the spatial window to gather (default: 3).
    """

    def __init__(self, hidden_dim, window_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        # Projection for the gathered window features
        # Input: hidden_dim * window_size (concatenated neighbors)
        # Output: hidden_dim (compressed context vector h_pair)
        self.proj = nn.Linear(hidden_dim * window_size, hidden_dim)

        # Gating mechanism
        # Input: Concatenation of current state h_i and context h_pair
        # Output: Gate values z_ij
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h, adj):
        """
        Args:
            h (torch.Tensor): Hidden states from BiGRU. Shape (B, L, D).
            adj (torch.Tensor): Adjacency indices. Shape (B, L, W).
                                Values are indices of paired neighbors. -1 indicates padding/unpaired.

        Returns:
            torch.Tensor: Updated hidden states. Shape (B, L, D).
        """
        B, L, D = h.shape

        # 1. Prepare for Gathering
        # We pad the hidden states with a zero vector at index L to handle -1 indices in adj.
        # padding: (B, 1, D)
        padding = torch.zeros(B, 1, D, device=h.device, dtype=h.dtype)
        h_padded = torch.cat([h, padding], dim=1)  # (B, L+1, D)

        # 2. Adjust Adjacency Indices
        # Replace -1 (unpaired/padding) with L (index of the zero vector)
        gather_idx = adj.clone()
        gather_idx[gather_idx == -1] = L

        # 3. Gather Windowed Context
        # We flatten the batch and length dimensions to use advanced indexing efficiently.
        # h_flat: (B * (L+1), D)
        h_flat = h_padded.view(B * (L + 1), D)

        # Calculate offsets to map indices to the flattened array
        # offset for batch b is b * (L+1)
        batch_offsets = torch.arange(B, device=h.device) * (L + 1)
        batch_offsets = batch_offsets.view(B, 1, 1)  # Broadcastable to (B, L, W)

        # Apply offsets
        final_gather_idx = gather_idx + batch_offsets
        final_gather_idx = final_gather_idx.view(-1)  # Flatten to (B*L*W)

        # Gather
        gathered = h_flat[final_gather_idx]  # (B*L*W, D)
        gathered = gathered.view(B, L, self.window_size, D)

        # 4. Local Aggregation
        # Flatten the window dimension: (B, L, W*D)
        gathered_flat = gathered.view(B, L, self.window_size * D)

        # Project to get the structural context vector h_pair
        h_pair = self.proj(gathered_flat)  # (B, L, D)

        # 5. Gated Injection
        # Compute gate z based on h_i and h_pair
        concat = torch.cat([h, h_pair], dim=-1)  # (B, L, 2D)
        z = torch.sigmoid(self.gate(concat))

        # 6. Residual Update with Masking
        # We only update positions that are actually paired.
        # The center of the window (index window_size // 2) corresponds to the direct pair.
        # If the direct pair index is -1, the base is unpaired.
        center_idx = self.window_size // 2
        mask = (adj[:, :, center_idx] != -1).unsqueeze(-1).float()  # (B, L, 1)

        # h' = h + mask * (z * h_pair)
        h_new = h + mask * (z * h_pair)

        return h_new


class RISRBiGRU(nn.Module):
    """
    Robust Iterative Structural-Refinement BiGRU (RISR-BiGRU).

    Architecture:
    1. 1D Convolutional Stem (Local feature extraction)
    2. Iterative Backbone: Stack of BiGRU blocks.
       - Blocks 1 to N-1: BiGRU -> Dropout -> WindowedInteractionModule
       - Block N: BiGRU -> Dropout
    3. Linear Output Head
    """

    def __init__(self, config: Config):
        super().__init__()

        # =====================================================================
        # 1. Convolutional Stem
        # =====================================================================
        # Projects sparse one-hot inputs (14 channels) to dense embedding space (256)
        # and aggregates local k-mers (kernel size 3).
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=config.input_channels,
                out_channels=config.conv_filters,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size // 2,
            ),
            nn.GELU(),
        )

        # =====================================================================
        # 2. Iterative Refinement Backbone
        # =====================================================================
        self.num_layers = config.num_layers
        self.gru_blocks = nn.ModuleList()
        self.interaction_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # BiGRU Hidden Dimension
        gru_hidden = config.hidden_dim
        # BiGRU Output Dimension (Bidirectional = 2 * hidden)
        self.backbone_dim = gru_hidden * 2

        current_dim = config.conv_filters

        for i in range(self.num_layers):
            # BiGRU Layer
            # Note: batch_first=True is standard
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=gru_hidden,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_blocks.append(gru)

            # Dropout
            self.dropouts.append(nn.Dropout(config.dropout))

            # Windowed Interaction Module
            # Added to all blocks except the final one
            if i < self.num_layers - 1:
                interaction = WindowedInteractionModule(
                    hidden_dim=self.backbone_dim, window_size=config.window_size
                )
                self.interaction_blocks.append(interaction)

            # Input dimension for next block is the output of current BiGRU
            current_dim = self.backbone_dim

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(self.backbone_dim, config.num_targets)

    def forward(self, x, adjacency):
        """
        Args:
            x (torch.Tensor): Input features. Shape (B, L, 14).
            adjacency (torch.Tensor): Adjacency map. Shape (B, L, W).

        Returns:
            torch.Tensor: Predictions. Shape (B, L, 5).
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Permute back for GRU: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Backbone
        for i in range(self.num_layers):
            # BiGRU
            # GRU returns (output, h_n), we only need output
            x, _ = self.gru_blocks[i](x)

            # Dropout
            x = self.dropouts[i](x)

            # Interaction Module (if applicable)
            if i < len(self.interaction_blocks):
                x = self.interaction_blocks[i](x, adjacency)

        # Head
        out = self.head(x)

        return out
