import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ConvStem(nn.Module):
    """
    1D Convolutional Stem to project sparse one-hot inputs into dense embedding space.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Length, Channels)
        Returns:
            Tensor of shape (Batch, Length, Out_Channels)
        """
        # Permute to (Batch, Channels, Length) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        # Permute back to (Batch, Length, Channels)
        return x.permute(0, 2, 1)


class InteractionModule(nn.Module):
    """
    Internally-Normalized Channel-Gated Structural Interaction Module.

    This module implements the stabilized structural injection mechanism:
    1. Gathers neighbor states based on secondary structure indices.
    2. Masks unpaired neighbors (zero-masking).
    3. Computes a non-linear message from the neighbor.
    4. Computes channel-wise gates using Internal Normalization to prevent saturation.
    5. Updates the current state via a residual connection.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message generation: Transform neighbor state
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism
        # Input: Concatenation of [h_i; h_j] -> 2 * hidden_dim
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Internal Normalization for the gate logits (Critical Stability Fix)
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # Final Output Normalization (Post-Norm)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.act = nn.GELU()

    def forward(self, h, bpps_indices, bpps_mask):
        """
        Args:
            h: Hidden states from BiGRU. Shape (B, L, D).
            bpps_indices: Indices of paired bases. Shape (B, L).
            bpps_mask: Mask for paired bases (1.0 if paired, 0.0 else). Shape (B, L, 1).
        """
        B, L, D = h.shape

        # 1. Gather neighbor states h_j
        # Expand indices to match feature dimension: (B, L, D)
        # bpps_indices are long tensors in [0, L-1].
        idx_expanded = bpps_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather along sequence dimension (dim=1)
        # Result: h_j[b, i, :] = h[b, bpps_indices[b, i], :]
        h_j = torch.gather(h, 1, idx_expanded)

        # 2. Zero-Masking
        # If unpaired, force h_j to be zero vector.
        # bpps_mask broadcasts to (B, L, D).
        h_j = h_j * bpps_mask

        # 3. Message Computation
        # m_ij = GELU(W_msg * h_j)
        m_ij = self.act(self.msg_proj(h_j))

        # 4. Internally-Normalized Channel-Gating
        # Concatenate h_i and h_j: (B, L, 2*D)
        cat_input = torch.cat([h, h_j], dim=-1)

        # Compute raw logits: z_ij
        z_ij = self.gate_proj(cat_input)

        # Apply LayerNorm to logits BEFORE Sigmoid
        # This prevents the vanishing gradient / saturation problem.
        z_norm = self.gate_norm(z_ij)

        # Compute Gate: g_ij = sigmoid(LayerNorm(z_ij))
        g_ij = torch.sigmoid(z_norm)

        # 5. Injection (Residual Update)
        # h_res = h_i + g_ij * m_ij
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class DIN_CG_BiGRU(nn.Module):
    """
    Deep Internally-Normalized Channel-Gated BiGRU Model.

    Architecture:
    1. ConvStem (Input -> 256)
    2. 4 Blocks of [BiGRU(384) -> InteractionModule] (Interaction omitted in last block)
    3. Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.input_dim = Config.INPUT_DIM
        self.conv_filters = Config.CONV_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM  # 384
        self.gru_output_dim = self.hidden_dim * 2  # Bidirectional -> 768
        self.num_layers = Config.NUM_LAYERS  # 4
        self.dropout_rate = Config.DROPOUT

        # 1. Convolutional Stem
        self.stem = ConvStem(
            self.input_dim, self.conv_filters, kernel_size=Config.CONV_KERNEL_SIZE
        )

        # 2. Deep Backbone
        self.blocks = nn.ModuleList()

        for i in range(self.num_layers):
            # Determine input dimension for the GRU
            # Layer 0: Input from ConvStem (256)
            # Layer 1+: Input from previous BiGRU output (768)
            gru_input_dim = self.conv_filters if i == 0 else self.gru_output_dim

            # Bidirectional GRU
            gru = nn.GRU(
                input_size=gru_input_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )

            # Interaction Module
            # Applied to all blocks EXCEPT the final one
            if i < self.num_layers - 1:
                interaction = InteractionModule(self.gru_output_dim)
            else:
                interaction = None

            self.blocks.append(nn.ModuleDict({"gru": gru, "interaction": interaction}))

        # 3. Output Head
        self.head = nn.Linear(self.gru_output_dim, Config.NUM_TARGETS)

        # Regularization
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, features, bpps_indices, bpps_mask):
        """
        Args:
            features: (B, L, 14)
            bpps_indices: (B, L)
            bpps_mask: (B, L, 1)
        Returns:
            logits: (B, L, 5)
        """
        # Stem
        x = self.stem(features)  # (B, L, 256)
        x = self.dropout(x)

        # Backbone Blocks
        for block in self.blocks:
            gru = block["gru"]
            interaction = block["interaction"]

            # BiGRU Forward
            # x input: (B, L, input_dim)
            # x output: (B, L, 768)
            x, _ = gru(x)

            # Interaction (if present)
            if interaction is not None:
                x = interaction(x, bpps_indices, bpps_mask)
                x = self.dropout(x)

        # Output Head
        logits = self.head(x)  # (B, L, 5)

        return logits
