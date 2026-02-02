import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Channel-Gating Structural Interaction Module.

    Implements:
    1. Point-to-Point Gathering of neighbor features.
    2. Explicit Zero-Masking for unpaired bases.
    3. Bias-Driven Loop Refinement (via decoupled message bias).
    4. Stabilized MLP Gate with Internal Normalization.
    5. Post-Normalization for deep stability.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled Message Generation
        # m_ij = GELU(W_msg * h_j + b_msg)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stabilized MLP Gate
        # Input: [h_i; h_j] -> 2 * hidden_dim
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)  # Internal Normalization
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: Hidden states (Batch, SeqLen, HiddenDim)
            pair_indices: Indices of paired bases (Batch, SeqLen)
            pair_masks: Mask where 1=paired, 0=unpaired (Batch, SeqLen)
        """
        B, L, D = x.shape

        # 1. Gather Neighbor Features (h_j)
        # Expand indices to (B, L, D) for gathering across hidden dimension
        # pair_indices is (B, L). We need to gather along dim 1.
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, D)
        h_j = torch.gather(x, 1, idx)

        # 2. Input Zero-Masking
        # Explicitly force h_j = 0 for unpaired bases
        # pair_masks is (B, L) -> (B, L, 1)
        mask = pair_masks.unsqueeze(-1)
        h_j = h_j * mask

        # 3. Decoupled Message Calculation
        # m_ij = GELU(W_msg * h_j + b_msg)
        # For unpaired bases (h_j=0), m_ij = GELU(b_msg), serving as a learnable loop embedding.
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Stabilized MLP Gate
        # Concatenate self (h_i) and neighbor (h_j)
        cat_input = torch.cat([x, h_j], dim=-1)

        # Project -> Internal Norm -> GELU -> Project -> Sigmoid
        z_raw = self.gate_proj1(cat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection
        # h_res = h_i + g_ij * m_ij
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    Deep Stabilized Bias-Refined Decoupled BiGRU.

    Architecture:
    1. 1D Convolutional Stem (Projection & Local Aggregation).
    2. Deep Backbone (4 Layers):
       - Bidirectional GRU
       - Structural Interaction Module (Layers 1-3 only)
       - Dropout
    3. Linear Output Head.
    """

    def __init__(self):
        super().__init__()

        self.input_dim = Config.input_dim
        self.conv_filters = Config.conv_filters
        self.hidden_dim = Config.hidden_dim
        self.n_layers = Config.n_layers
        self.dropout_rate = Config.dropout
        self.num_classes = Config.num_classes

        # 1. Convolutional Stem
        self.conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_filters,
            kernel_size=Config.conv_kernel_size,
            padding=Config.conv_kernel_size // 2,
        )
        self.act = nn.GELU()

        # 2. Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_dim = self.conv_filters

        for i in range(self.n_layers):
            # BiGRU
            # We want the output dimension to be self.hidden_dim (384).
            # Since it's bidirectional, hidden_size must be half of that.
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=self.hidden_dim // 2,
                bidirectional=True,
                batch_first=True,
            )
            self.gru_layers.append(gru)
            self.dropouts.append(nn.Dropout(self.dropout_rate))

            # Interaction Module
            # Applied after GRU in blocks 0, 1, 2. Skipped in block 3 (final block).
            if i < self.n_layers - 1:
                self.interaction_layers.append(
                    StructuralInteractionModule(self.hidden_dim)
                )

            # Update current_dim for next layer (BiGRU output is always hidden_dim)
            current_dim = self.hidden_dim

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, inputs, pair_indices, pair_masks):
        """
        Args:
            inputs: (Batch, SeqLen, 14)
            pair_indices: (Batch, SeqLen)
            pair_masks: (Batch, SeqLen)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.transpose(1, 2)

        # Stem
        x = self.conv(x)
        x = self.act(x)

        # Permute back for GRU: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # Backbone
        for i in range(self.n_layers):
            # GRU
            x, _ = self.gru_layers[i](x)
            x = self.dropouts[i](x)

            # Interaction (if exists for this layer)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_masks)

        # Head
        out = self.head(x)

        return out
