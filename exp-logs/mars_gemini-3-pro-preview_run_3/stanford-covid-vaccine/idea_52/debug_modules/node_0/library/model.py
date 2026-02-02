import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Implements the Decoupled Channel-Gating with Bias-Driven Loop Refinement
    and Stabilized MLP Gating.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled Message Projection
        # m_ij = GELU(W_msg * h_j + b_msg)
        # For unpaired bases (h_j masked to 0), this becomes GELU(b_msg),
        # serving as a learnable loop embedding.
        self.w_msg = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Stabilized MLP Gate
        # Projects concatenated context [h_i; h_j] -> scalar gate
        # Uses internal LayerNorm to prevent saturation while preserving sparsity semantics.
        self.w_g1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.ln_gate = nn.LayerNorm(hidden_dim)
        self.w_g2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Post-Normalization for the residual block
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, paired_mask):
        """
        Args:
            h: (Batch, Seq_Len, Hidden_Dim)
            pair_indices: (Batch, Seq_Len) - Indices of paired bases
            paired_mask: (Batch, Seq_Len, 1) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, D = h.shape

        # 1. Gather Neighbor States (Point-to-Point)
        # Expand indices to match hidden dim: (B, L, D)
        flat_indices = pair_indices.view(B, L, 1).expand(-1, -1, D)
        # Gather h_j: For each position i, get h at index pair_indices[i]
        h_j = torch.gather(h, 1, flat_indices)

        # 2. Input Zero-Masking (Strictly avoid self-loops/noise)
        # If position i is unpaired, force h_j = 0.
        h_j = h_j * paired_mask

        # 3. Decoupled Message Calculation
        # m_ij = GELU(W_msg * h_j + b_msg)
        m_ij = F.gelu(self.w_msg(h_j))

        # 4. Stabilized MLP Gate Calculation
        # Joint Context: [h_i; h_j]
        # Note: We do NOT normalize the input [h_i; h_j] to preserve magnitude semantics.
        cat_input = torch.cat([h, h_j], dim=-1)

        # Project -> Internal Norm -> Act -> Project -> Sigmoid
        z_raw = self.w_g1(cat_input)
        z_norm = self.ln_gate(z_raw)  # Stabilizes the MLP internals
        z_act = F.gelu(z_norm)
        logits = self.w_g2(z_act)
        g_ij = torch.sigmoid(logits)  # No logit norm, allowing saturation

        # 5. Injection (Residual)
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        # Stabilizes the deep backbone
        h_out = self.ln_out(h_res)

        return h_out


class DeepStabilizedBiGRU(nn.Module):
    """
    4-Layer Bidirectional GRU with Interleaved Decoupled Post-Norm Structural Injection.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.input_dim = Config.INPUT_DIM  # 14
        self.hidden_dim = Config.HIDDEN_DIM  # 384
        self.num_layers = Config.NUM_LAYERS  # 4
        self.num_targets = Config.NUM_TARGETS  # 5
        self.conv_filters = Config.CONV_FILTERS  # 256
        self.kernel_size = Config.CONV_KERNEL_SIZE  # 3
        self.dropout_rate = Config.DROPOUT

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs to dense embedding and aggregates local k-mers.
        self.conv_stem = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.stem_act = nn.GELU()

        # Projection to match backbone hidden dimension (256 -> 384)
        self.stem_proj = nn.Linear(self.conv_filters, self.hidden_dim)

        # 2. Deep Stabilized Backbone
        # Consists of 4 Blocks.
        # Blocks 0, 1, 2: BiGRU -> Interaction Module
        # Block 3: BiGRU only (Final block)
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        for i in range(self.num_layers):
            # BiGRU: Output dim is hidden_dim * 2 if bidirectional, so we set hidden_size = dim // 2
            self.gru_layers.append(
                nn.GRU(
                    input_size=self.hidden_dim,
                    hidden_size=self.hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Add Interaction Module to all but the last layer
            if i < self.num_layers - 1:
                self.interaction_layers.append(
                    StructuralInteractionModule(self.hidden_dim)
                )

        self.dropout = nn.Dropout(self.dropout_rate)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, x, pair_indices):
        """
        Args:
            x: (Batch, Seq_Len, 14) - Input features
            pair_indices: (Batch, Seq_Len) - Structural pair indices
        """
        # --- Preprocessing ---
        # Generate Paired Mask from Input Features
        # Channel 6 corresponds to '.' (unpaired) in the one-hot encoding.
        # If x[..., 6] == 1, the base is unpaired.
        # Mask = 1 if paired, 0 if unpaired.
        unpaired_prob = x[:, :, 6]
        paired_mask = (unpaired_prob < 0.5).float().unsqueeze(-1)  # (B, L, 1)

        # --- Stem ---
        # Permute for Conv1d: (B, C, L)
        x_perm = x.permute(0, 2, 1)
        x_conv = self.conv_stem(x_perm)
        x_conv = self.stem_act(x_conv)

        # Permute back: (B, L, C)
        x_emb = x_conv.permute(0, 2, 1)

        # Project to backbone dimension
        h = self.stem_proj(x_emb)

        # --- Backbone ---
        for i in range(self.num_layers):
            # 1. BiGRU
            # h comes in as (B, L, hidden_dim)
            # GRU out is (B, L, hidden_dim)
            h_gru, _ = self.gru_layers[i](h)

            # Update state
            h = h_gru

            # 2. Structural Interaction (if applicable)
            if i < len(self.interaction_layers):
                h = self.interaction_layers[i](h, pair_indices, paired_mask)

            # 3. Dropout (between blocks)
            if i < self.num_layers - 1:
                h = self.dropout(h)

        # --- Head ---
        out = self.head(h)

        return out
