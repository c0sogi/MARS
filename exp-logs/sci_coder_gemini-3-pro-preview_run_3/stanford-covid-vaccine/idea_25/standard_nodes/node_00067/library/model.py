import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralGatingBlock(nn.Module):
    """
    Implements the Pre-Norm Structural Interaction Module with Zero-Masked Channel-Gating.

    Logic:
    1. Pre-Norm: h_norm = LayerNorm(h_in)
    2. Gather: Retrieve neighbor state h_j based on bpp_indices.
    3. Zero-Mask: If unpaired, force h_j to 0.
    4. Message: m_ij = GELU(W_msg * h_j)
    5. Gate: g_ij = Sigmoid(W_gate * [h_i; h_j])
    6. Update: h_out = h_in + g_ij * m_ij
    """

    def __init__(self, dim):
        super(StructuralGatingBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bpp_indices):
        """
        Args:
            x: Tensor of shape (Batch, SeqLen, Dim)
            bpp_indices: LongTensor of shape (Batch, SeqLen) containing paired indices or -1.
        """
        batch_size, seq_len, dim = x.shape

        # 1. Pre-Normalization
        h_norm = self.norm(x)

        # 2. Gather Neighbor States
        # Handle -1 indices by replacing with 0 temporarily.
        # We will mask the result afterwards, so the value at index 0 doesn't matter for unpaired bases.
        mask = (bpp_indices != -1).unsqueeze(-1).type_as(h_norm)  # (B, L, 1)
        safe_indices = bpp_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices for gather: (B, L, Dim)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, dim)

        # Gather: h_neighbor[b, i, :] = h_norm[b, safe_indices[b, i], :]
        h_neighbor = torch.gather(h_norm, 1, gather_indices)

        # 3. Zero-Masking for Unpaired Bases
        # Explicitly force neighbor vector to 0 if no pair exists
        h_neighbor = h_neighbor * mask

        # 4. Non-Linear Message
        m = self.act(self.msg_proj(h_neighbor))

        # 5. Channel-Wise Gating
        # Concatenate current state (h_norm) and neighbor state (h_neighbor)
        cat_feat = torch.cat([h_norm, h_neighbor], dim=-1)
        g = self.sigmoid(self.gate_proj(cat_feat))

        # 6. Residual Injection
        out = x + g * m

        return out


class RNAModel(nn.Module):
    """
    Deep Pre-Norm BiGRU with Zero-Masked Structural Channel-Gating.

    Architecture:
    - Input: (N, L, 14)
    - Stem: Conv1d -> GELU
    - Backbone: 4 Layers of [BiGRU -> StructuralGatingBlock]
    - Head: Linear -> 5 targets
    """

    def __init__(self, config):
        super(RNAModel, self).__init__()

        self.input_dim = config.input_dim
        self.conv_filters = config.conv_filters
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_targets = config.num_targets
        self.kernel_size = config.conv_kernel_size

        # Convolutional Stem
        # Projects sparse one-hot inputs to dense embedding and aggregates local k-mers
        self.conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.stem_act = nn.GELU()

        # Backbone
        # BiGRU output dimension is hidden_dim * 2 (for bidirectional)
        self.gru_out_dim = self.hidden_dim * 2

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Input dim is conv_filters for the first layer, else gru_out_dim from previous block
            input_size = self.conv_filters if i == 0 else self.gru_out_dim

            gru = nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )

            gating = StructuralGatingBlock(self.gru_out_dim)

            self.layers.append(nn.ModuleDict({"gru": gru, "gating": gating}))

        # Output Head
        self.head = nn.Linear(self.gru_out_dim, self.num_targets)

    def forward(self, inputs, bpp_indices):
        """
        Args:
            inputs: (Batch, SeqLen, InputDim)
            bpp_indices: (Batch, SeqLen)
        Returns:
            outputs: (Batch, SeqLen, NumTargets)
        """
        # Permute for Conv1d: (B, C, L)
        x = inputs.transpose(1, 2)
        x = self.stem_act(self.conv(x))
        x = x.transpose(1, 2)  # Back to (B, L, C)

        # Pass through backbone layers
        for layer in self.layers:
            gru = layer["gru"]
            gating = layer["gating"]

            # BiGRU
            x, _ = gru(x)

            # Structural Gating
            x = gating(x, bpp_indices)

        # Prediction Head
        out = self.head(x)

        return out
