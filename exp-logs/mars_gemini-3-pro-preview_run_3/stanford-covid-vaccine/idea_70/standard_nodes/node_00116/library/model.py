import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLUInteractionModule(nn.Module):
    """
    GLU-Refined Decoupled Interaction Module.

    Implements the structural injection logic:
    1. Point-to-Point Gather of neighbor features (h_j).
    2. Input Zero-Masking for unpaired bases.
    3. GLU-based Message Passing: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
    4. Wide Stabilized MLP Gate for injection control.
    5. Additive Injection with Post-Normalization.
    """

    def __init__(self, hidden_dim, gate_hidden_dim):
        super().__init__()

        # Message Generation: GLU
        # Projects neighbor state to content and gate components
        self.w_c = nn.Linear(hidden_dim, hidden_dim)
        self.w_g = nn.Linear(hidden_dim, hidden_dim)

        # Wide Stabilized MLP Gate
        # Input: Concatenation of [h_i; h_j] -> 2 * hidden_dim
        # Projects to a wide dimension to avoid bottlenecks
        self.w_in = nn.Linear(hidden_dim * 2, gate_hidden_dim)
        self.layer_norm_internal = nn.LayerNorm(gate_hidden_dim)
        self.w_out = nn.Linear(gate_hidden_dim, hidden_dim)

        # Post-Normalization for the residual block
        self.layer_norm_post = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Tensor (Batch, Seq_Len, Hidden_Dim)
            pair_indices: LongTensor (Batch, Seq_Len) - Indices of paired bases
            pair_mask: FloatTensor (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, D = x.shape

        # 1. Gather Neighbor Features (h_j)
        # Create batch indices grid: (B, L)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, L)

        # Gather h_j using advanced indexing.
        # Note: pair_indices contains 0 for unpaired bases (dummy index),
        # but these will be masked out in the next step.
        x_neighbor = x[batch_idx, pair_indices]  # (B, L, D)

        # 2. Input Zero-Masking
        # Explicitly force h_j = 0 for unpaired bases.
        # This ensures the GLU bias term acts as a learnable "loop embedding" for unpaired positions.
        mask_expanded = pair_mask.unsqueeze(-1)  # (B, L, 1)
        x_neighbor = x_neighbor * mask_expanded

        # 3. GLU Message Calculation
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        msg_content = self.w_c(x_neighbor)
        msg_gate = torch.sigmoid(self.w_g(x_neighbor))
        m_ij = msg_content * msg_gate

        # 4. Wide Stabilized MLP Gate
        # Input: Concatenation of current state h_i and neighbor state h_j
        concat_input = torch.cat([x, x_neighbor], dim=-1)  # (B, L, 2*D)

        # Wide Projection -> LayerNorm -> GELU -> Projection -> Sigmoid
        z_wide = self.w_in(concat_input)
        z_norm = self.layer_norm_internal(z_wide)
        z_act = F.gelu(z_norm)
        logits = self.w_out(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection & Post-Normalization
        # Additive injection: h_res = h_i + g_ij * m_ij
        h_res = x + g_ij * m_ij
        h_out = self.layer_norm_post(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity GLU-Refined Decoupled BiGRU Model.

    Architecture:
    - Input: One-hot features (14 channels)
    - Stem: 1D Convolution
    - Backbone: 4 Layers of Bidirectional GRU (High Capacity)
    - Interaction: GLUInteractionModule interleaved between GRU layers (except final)
    - Head: Linear projection to targets
    """

    def __init__(self, config=Config):
        super().__init__()

        self.input_dim = config.input_dim
        self.stem_channels = config.stem_channels
        # BiGRU output dimension is hidden_dim * 2 (bidirectional)
        self.hidden_dim = config.hidden_dim * 2
        self.gate_hidden_dim = config.gate_hidden_dim
        self.num_layers = config.num_layers
        self.num_targets = config.num_targets
        self.dropout_p = config.dropout

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs to dense embedding space
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.stem_channels,
                kernel_size=config.stem_kernel_size,
                padding=config.stem_kernel_size // 2,
            ),
            nn.GELU(),
        )

        # 2. Backbone Blocks
        self.blocks = nn.ModuleList()

        for i in range(self.num_layers):
            block = nn.ModuleDict()

            # Determine input size for GRU
            # Layer 0 takes stem output, subsequent layers take previous GRU output
            gru_input_size = self.stem_channels if i == 0 else self.hidden_dim

            block["gru"] = nn.GRU(
                input_size=gru_input_size,
                hidden_size=config.hidden_dim,  # 384 per direction
                batch_first=True,
                bidirectional=True,
            )

            block["dropout"] = nn.Dropout(self.dropout_p)

            # Add Interaction Module to all blocks EXCEPT the final one
            # Layers 0, 1, 2 get interaction. Layer 3 does not.
            if i < self.num_layers - 1:
                block["interaction"] = GLUInteractionModule(
                    self.hidden_dim, self.gate_hidden_dim
                )

            self.blocks.append(block)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim)
            pair_indices: (Batch, Seq_Len)
            pair_mask: (Batch, Seq_Len)
        """

        # Stem Processing
        # Permute for Conv1d: (B, C, L)
        x = x.transpose(1, 2)
        x = self.stem(x)
        # Permute back: (B, L, C)
        x = x.transpose(1, 2)

        # Backbone Processing
        for block in self.blocks:
            # BiGRU
            x, _ = block["gru"](x)

            # Dropout
            x = block["dropout"](x)

            # Interaction (if exists in this block)
            if "interaction" in block:
                x = block["interaction"](x, pair_indices, pair_mask)

        # Output Head
        out = self.head(x)

        return out
