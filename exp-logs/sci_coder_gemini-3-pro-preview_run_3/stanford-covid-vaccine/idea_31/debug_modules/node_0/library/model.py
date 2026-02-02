import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InteractionModule(nn.Module):
    """
    Internally-Normalized Channel-Gated Structural Interaction Module.

    Implements:
    1. Point-to-Point Gather of paired states (h_j).
    2. Zero-Masking for unpaired bases.
    3. Non-Linear Message transformation (m_ij).
    4. Internally-Normalized Gating (LayerNorm on logits).
    5. Residual Connection with Post-Normalization.
    """

    def __init__(self, hidden_dim):
        super(InteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Message transformation: h_j -> m_ij
        self.w_msg = nn.Linear(hidden_dim, hidden_dim)

        # Gate transformation: [h_i; h_j] -> z_ij
        # Input dimension is 2 * hidden_dim because we concat h_i and h_j
        self.w_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        # Normalization layers
        # gate_norm: Applied to logits before sigmoid to prevent saturation
        self.gate_norm = nn.LayerNorm(hidden_dim)
        # out_norm: Applied after residual connection (Post-Norm)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (N, L, C) representing hidden states h_i
            pair_indices: LongTensor of shape (N, L) with indices of paired bases (-1 if unpaired)
        """
        N, L, C = x.shape

        # 1. Prepare indices and mask
        # Create mask for valid pairs: 1 if paired, 0 if unpaired (-1)
        # pair_indices can be on GPU, so we ensure mask matches device/type
        mask = (pair_indices != -1).unsqueeze(-1).type_as(x)  # (N, L, 1)

        # Replace -1 with 0 to allow valid gather (values will be masked out later)
        safe_indices = pair_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices for gather: (N, L, C)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, C)

        # 2. Gather h_j
        # x is (N, L, C), gather along dim 1 (sequence length)
        h_j = torch.gather(x, 1, gather_indices.long())

        # 3. Masking: Force zero vector if unpaired (strictly avoid self-loops/noise)
        h_j = h_j * mask

        # 4. Non-Linear Message
        m = F.gelu(self.w_msg(h_j))

        # 5. Internally-Normalized Gate
        # Concatenate h_i (x) and h_j
        cat_input = torch.cat([x, h_j], dim=-1)  # (N, L, 2C)
        z = self.w_gate(cat_input)

        # Apply LayerNorm to logits before Sigmoid (Critical Fix for stability)
        z_norm = self.gate_norm(z)
        g = torch.sigmoid(z_norm)

        # 6. Injection (Residual Update)
        res = x + g * m

        # 7. Post-Normalization (Critical Fix for deep stacking)
        out = self.out_norm(res)

        return out


class SDCGBiGRU(nn.Module):
    """
    Stabilized Deep Channel-Gated BiGRU (SDCG-BiGRU).

    Architecture:
    1. 1D Convolutional Stem (Projects sparse one-hot to dense embedding).
    2. 4 Blocks of BiGRU + Interaction Module (interleaved).
       - Block 1-3: BiGRU -> Interaction
       - Block 4: BiGRU -> None (as per strategy)
    3. Final Linear Head.
    """

    def __init__(self):
        super(SDCGBiGRU, self).__init__()

        # Configuration
        self.input_channels = Config.INPUT_CHANNELS
        self.stem_channels = Config.STEM_CHANNELS
        self.hidden_dim = Config.HIDDEN_DIM  # BiGRU hidden size per direction
        self.num_layers = Config.NUM_LAYERS
        self.dropout_p = Config.DROPOUT
        self.num_targets = 5

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.stem_channels,
                kernel_size=Config.STEM_KERNEL_SIZE,
                padding=Config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )

        # 2. Deep Backbone
        self.blocks = nn.ModuleList()

        # Track dimensions
        # Stem output is (N, stem_channels, L) -> permuted to (N, L, stem_channels)
        current_input_dim = self.stem_channels

        for i in range(self.num_layers):
            # BiGRU Layer
            # Input: current_input_dim
            # Hidden: self.hidden_dim (per direction)
            # Output: 2 * self.hidden_dim (Concatenated directions)
            gru = nn.GRU(
                input_size=current_input_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )

            gru_out_dim = 2 * self.hidden_dim

            # Interaction Module
            # Applied in all blocks except the final one
            if i < self.num_layers - 1:
                interaction = InteractionModule(gru_out_dim)
                block = nn.ModuleList([gru, interaction])
            else:
                block = nn.ModuleList([gru, None])

            self.blocks.append(block)

            # Next layer input is current layer output
            current_input_dim = gru_out_dim

        # Dropout
        self.dropout = nn.Dropout(self.dropout_p)

        # 3. Output Head
        self.head = nn.Linear(current_input_dim, self.num_targets)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Input tensor (N, L, 14)
            pair_indices: Structural indices (N, L)
        Returns:
            out: Prediction tensor (N, L, 5)
        """
        # Permute for Conv1d: (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)

        # Apply Stem
        x = self.stem(x)

        # Permute back for RNN: (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1)

        # Apply Backbone Blocks
        for gru, interaction in self.blocks:
            # BiGRU
            x, _ = gru(x)

            # Interaction Module (if present)
            if interaction is not None:
                x = interaction(x, pair_indices)

            # Dropout
            if self.dropout_p > 0:
                x = self.dropout(x)

        # Apply Head
        out = self.head(x)

        return out
