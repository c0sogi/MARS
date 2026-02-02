import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InteractionModule(nn.Module):
    """
    Stabilized Decoupled Interaction Module.

    Implements the structural injection mechanism with:
    1. Point-to-point gathering of paired states.
    2. Zero-masking for unpaired bases (allowing bias-driven loop refinement).
    3. Internal Gate Normalization to prevent saturation.
    4. Post-Normalization for deep network stability.
    """

    def __init__(self, hidden_dim):
        super(InteractionModule, self).__init__()

        # Message generation: Projects h_pair to message space
        # Bias is crucial here as it acts as the 'loop embedding' when input is masked to 0
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Gating mechanism
        # Input: Concatenation of [h_i, h_pair] -> 2 * hidden_dim
        self.gate_proj1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)  # Internal Gate Normalization
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-injection normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Tensor (Batch, Seq, Hidden)
            pair_indices: Tensor (Batch, Seq) - Indices of paired bases
            pair_mask: Tensor (Batch, Seq) - 1.0 if paired, 0.0 if unpaired
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. Gather paired states
        # Create batch indices: (B, L)
        batch_idx = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Gather: x[b, pair_indices[b, i], :]
        # This retrieves h_j for every i. If i is paired with j, we get h_j.
        # If i is unpaired, pair_indices[i] is 0 (safe index), but we mask it next.
        x_pair = x[batch_idx, pair_indices]  # Shape: (B, L, D)

        # 2. Input Zero-Masking
        # If unpaired, h_j should be strictly 0.
        # pair_mask is (B, L). Expand to (B, L, 1)
        mask = pair_mask.unsqueeze(-1)
        x_pair = x_pair * mask

        # 3. Decoupled Message (Bias-Refined)
        # m_ij = GELU(W * h_j + b)
        # If h_j is 0 (unpaired), m_ij = GELU(b), serving as a learnable loop embedding.
        m = self.act(self.msg_proj(x_pair))

        # 4. Stabilized MLP Gate
        # Concatenate h_i and h_j (masked)
        cat_input = torch.cat([x, x_pair], dim=-1)  # (B, L, 2*D)

        # Project
        z_raw = self.gate_proj1(cat_input)

        # Internal Normalization (Stabilizes the gate internals)
        z_norm = self.gate_norm(z_raw)

        # Activation
        z_act = self.act(z_norm)

        # Logit Projection and Sigmoid (No Logit Norm)
        logits = self.gate_proj2(z_act)
        g = self.sigmoid(logits)

        # 5. Injection
        h_res = x + g * m

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class HC_SDBR_BiGRU(nn.Module):
    """
    High-Capacity Stabilized Decoupled Bias-Refined BiGRU.

    Architecture:
    1. 1D CNN Stem
    2. 4 Layers of BiGRU (Hidden 384x2 = 768)
    3. Interaction Modules interleaved (except after last block)
    4. Linear Head
    """

    def __init__(self):
        super(HC_SDBR_BiGRU, self).__init__()

        # Dimensions
        input_channels = Config.INPUT_CHANNELS
        cnn_filters = Config.CNN_FILTERS
        hidden_dim = Config.HIDDEN_DIM  # 384
        total_hidden = hidden_dim * 2  # 768
        num_layers = Config.NUM_LAYERS  # 4
        num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        # Projects sparse inputs into a dense embedding space
        self.stem_conv = nn.Conv1d(
            input_channels, cnn_filters, kernel_size=Config.CNN_KERNEL_SIZE, padding=1
        )
        self.stem_act = nn.GELU()

        # Project stem output to backbone capacity (256 -> 768)
        self.stem_proj = nn.Linear(cnn_filters, total_hidden)

        # 2. Backbone Blocks
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        for i in range(num_layers):
            # BiGRU Layer
            # Input size is total_hidden because we project stem and previous layers output total_hidden
            gru = nn.GRU(
                input_size=total_hidden,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Interaction Module
            # Added to all blocks except the final one as per strategy description
            if i < num_layers - 1:
                self.interaction_layers.append(InteractionModule(total_hidden))
            else:
                self.interaction_layers.append(None)

        self.dropout = nn.Dropout(Config.DROPOUT)

        # 3. Output Head
        self.head = nn.Linear(total_hidden, num_targets)

    def forward(self, inputs, pair_indices, pair_mask):
        """
        Args:
            inputs: (B, L, 14)
            pair_indices: (B, L)
            pair_mask: (B, L)
        """
        # 1. Stem
        # Conv1d expects (B, Channels, Length)
        x = inputs.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        x = x.permute(0, 2, 1)  # Back to (B, L, C)

        # Project to backbone dimension
        x = self.stem_proj(x)

        # 2. Backbone
        for i, gru in enumerate(self.gru_layers):
            # GRU
            x, _ = gru(x)

            # Dropout
            x = self.dropout(x)

            # Interaction (if present for this layer)
            interaction = self.interaction_layers[i]
            if interaction is not None:
                x = interaction(x, pair_indices, pair_mask)

        # 3. Head
        out = self.head(x)

        return out
