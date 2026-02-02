import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralGatingUnit(nn.Module):
    """
    Implements the Post-Norm Structural Interaction Module.

    Mechanism:
    1. Point-to-Point Gather: Retrieve neighbor state h_j using bpp_indices.
    2. Zero-Masking: Force h_j to zero if the base is unpaired (bpp_mask).
    3. Non-Linear Message: m_ij = GELU(W_msg * h_j).
    4. Channel-Gating: g_ij = Sigmoid(W_gate * [h_i; h_j]).
    5. Residual Injection: h_res = h_i + g_ij * m_ij.
    6. Post-Normalization: h_out = LayerNorm(h_res).
    """

    def __init__(self, hidden_dim):
        super(StructuralGatingUnit, self).__init__()
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        # Input to gate is concatenation of h_i and h_j, so 2 * hidden_dim
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, bpp_indices, bpp_mask):
        """
        Args:
            h: Hidden states (Batch, Seq_Len, Hidden_Dim)
            bpp_indices: Adjacency indices (Batch, Seq_Len)
            bpp_mask: Mask indicating paired status (Batch, Seq_Len)
        """
        N, L, C = h.shape

        # 1. Gather neighbors
        # Expand indices to (N, L, C) to gather along the feature dimension
        gather_indices = bpp_indices.unsqueeze(-1).expand(-1, -1, C)
        h_neighbor = torch.gather(h, 1, gather_indices)  # (N, L, C)

        # 2. Zero-Masking
        # bpp_mask is (N, L). 1.0 if paired, 0.0 if unpaired.
        # Expand to (N, L, 1) for broadcasting
        mask = bpp_mask.unsqueeze(-1)
        h_neighbor = h_neighbor * mask

        # 3. Non-Linear Message
        m = F.gelu(self.msg_proj(h_neighbor))

        # 4. Channel-Wise Gating
        # Concatenate current state h and neighbor state h_neighbor
        cat_feat = torch.cat([h, h_neighbor], dim=-1)  # (N, L, 2*C)
        g = torch.sigmoid(self.gate_proj(cat_feat))

        # 5. Residual Injection
        h_res = h + g * m

        # 6. Post-Normalization
        out = self.norm(h_res)

        return out


class RNARegressor(nn.Module):
    """
    Stabilized Deep Zero-Masked Channel-Gated BiGRU Architecture.

    Components:
    - Convolutional Stem (1D Conv, 256 filters)
    - 4-Layer Backbone (BiGRU + StructuralGatingUnit)
    - Linear Output Head
    """

    def __init__(self, config):
        super(RNARegressor, self).__init__()
        self.config = config

        # =====================================================================
        # 1. Convolutional Stem
        # =====================================================================
        # Projects sparse one-hot inputs (14 channels) to dense embedding (256 channels)
        self.conv_stem = nn.Conv1d(
            in_channels=config.input_dim,
            out_channels=256,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.stem_act = nn.GELU()

        # =====================================================================
        # 2. Deep Stabilized Backbone
        # =====================================================================
        self.blocks = nn.ModuleList()

        # We want the output of the BiGRU to match config.hidden_dim (384).
        # Since it's bidirectional, we set hidden_size = config.hidden_dim // 2.
        gru_hidden_size = config.hidden_dim // 2

        # Input dimension for the first RNN layer comes from the Stem (256).
        input_dim = 256

        for i in range(config.num_layers):
            # Bidirectional GRU
            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=gru_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

            # Structural Interaction Module
            # Applied after every block EXCEPT the final one.
            if i < config.num_layers - 1:
                interaction = StructuralGatingUnit(config.hidden_dim)
            else:
                interaction = None

            self.blocks.append(nn.ModuleList([gru, interaction]))

            # The output of the BiGRU is config.hidden_dim (384),
            # which becomes the input for the next layer.
            input_dim = config.hidden_dim

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(config.hidden_dim, config.num_targets)

    def forward(self, inputs, bpp_indices, bpp_mask):
        """
        Args:
            inputs: (Batch, Seq_Len, 14)
            bpp_indices: (Batch, Seq_Len)
            bpp_mask: (Batch, Seq_Len)
        """
        # --- Stem ---
        # Permute to (N, C, L) for Conv1d
        x = inputs.transpose(1, 2)
        x = self.conv_stem(x)
        x = self.stem_act(x)
        # Permute back to (N, L, C) for RNN
        x = x.transpose(1, 2)

        # --- Backbone ---
        for gru, interaction in self.blocks:
            # BiGRU Pass
            # x shape: (N, L, Input_Dim) -> (N, L, Hidden_Dim)
            x, _ = gru(x)

            # Structural Interaction (if present)
            if interaction is not None:
                x = interaction(x, bpp_indices, bpp_mask)

        # --- Head ---
        # x shape: (N, L, 384) -> (N, L, 5)
        out = self.head(x)

        return out
