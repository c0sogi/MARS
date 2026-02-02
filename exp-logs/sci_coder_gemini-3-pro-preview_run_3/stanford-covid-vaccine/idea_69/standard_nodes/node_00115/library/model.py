import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLUInteractionModule(nn.Module):
    """
    GLU-Refined Decoupled Structural Interaction Module.

    Features:
    - Point-to-Point Gather of neighbor features.
    - Zero-Masking for unpaired bases.
    - GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
    - Wide Stabilized MLP Gate for injection control.
    - Additive Injection with Post-Normalization.
    """

    def __init__(self, hidden_dim, wide_dim, dropout=0.1):
        super(GLUInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # GLU Message Projection: Maps h_j to (Content + Gate) components
        # Output dim is 2 * hidden_dim to support GLU split
        self.message_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        # Wide Stabilized MLP Gate
        # Input: Concatenation of [h_i, h_j] -> 2 * hidden_dim
        self.gate_in_proj = nn.Linear(hidden_dim * 2, wide_dim)
        self.gate_norm = nn.LayerNorm(wide_dim)
        self.gate_out_proj = nn.Linear(wide_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_masks):
        """
        Args:
            h: (Batch, Seq, Hidden) - Input hidden states.
            pair_indices: (Batch, Seq) - Indices of paired bases.
            pair_masks: (Batch, Seq) - 1.0 if paired, 0.0 if unpaired.
        """
        B, L, D = h.shape

        # 1. Gather Neighbor Features (h_j)
        # Expand indices to (B, L, D) for gathering along sequence dimension
        gather_idx = pair_indices.unsqueeze(-1).expand(-1, -1, D)
        h_j = torch.gather(h, 1, gather_idx)  # (B, L, D)

        # 2. Mask Unpaired Neighbors
        # Ensure h_j is strictly 0 vector for unpaired bases
        mask = pair_masks.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. GLU Message Calculation
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired bases (h_j=0), this results in learnable bias interaction
        msg_raw = self.message_proj(h_j)
        msg_content, msg_gate = torch.chunk(msg_raw, 2, dim=-1)
        m_ij = msg_content * torch.sigmoid(msg_gate)

        # 4. Wide Stabilized MLP Gate
        # Determine how much of the message to inject based on context [h_i, h_j]
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2*D)

        z = self.gate_in_proj(cat_input)  # Project to Wide Dim
        z = self.gate_norm(z)  # Internal Normalization
        z = F.gelu(z)  # Activation
        logits = self.gate_out_proj(z)  # Project back to D
        g_ij = torch.sigmoid(logits)  # Sigmoid for gating (0 to 1)

        # 5. Injection and Post-Normalization
        # Additive injection: h_new = h + gate * message
        h_res = h + g_ij * m_ij
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity GLU-Refined Decoupled BiGRU Model.

    Architecture:
    1. 1D Convolutional Stem (Input -> Embedding).
    2. 4-Layer Bidirectional GRU Backbone.
    3. GLU-Refined Interaction Modules interleaved after layers 0, 1, 2.
    4. Linear Output Head.
    """

    def __init__(self, config=Config):
        super(RNAModel, self).__init__()

        # --- 1. Convolutional Stem ---
        # Projects sparse one-hot features to dense embedding space
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=config.INPUT_DIM,
                out_channels=config.STEM_FILTERS,
                kernel_size=config.STEM_KERNEL_SIZE,
                padding=config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )

        # --- 2. Backbone & Interactions ---
        self.layers = nn.ModuleList()
        self.interactions = nn.ModuleList()
        self.dropout = nn.Dropout(config.DROPOUT)

        # BiGRU Output Dimension = Hidden * 2
        gru_out_dim = config.HIDDEN_DIM * 2

        # Input dimension for the first GRU layer comes from Stem
        current_input_dim = config.STEM_FILTERS

        for i in range(config.NUM_LAYERS):
            # Bidirectional GRU Layer
            gru = nn.GRU(
                input_size=current_input_dim,
                hidden_size=config.HIDDEN_DIM,
                batch_first=True,
                bidirectional=True,
            )
            self.layers.append(gru)

            # Interaction Module (Applied after layers 0, 1, 2; skipped after last layer)
            if i < config.NUM_LAYERS - 1:
                inter = GLUInteractionModule(
                    hidden_dim=gru_out_dim,
                    wide_dim=config.GATE_WIDE_DIM,
                    dropout=config.DROPOUT,
                )
                self.interactions.append(inter)

            # Update input dimension for next layer (output of BiGRU is hidden*2)
            current_input_dim = gru_out_dim

        # --- 3. Output Head ---
        self.head = nn.Linear(gru_out_dim, config.NUM_TARGETS)

    def forward(self, inputs, pair_indices, pair_masks):
        """
        Args:
            inputs: (Batch, Seq, 14) - One-hot encoded features.
            pair_indices: (Batch, Seq) - Indices of paired bases.
            pair_masks: (Batch, Seq) - Mask for paired bases.
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.transpose(1, 2)

        # Stem
        x = self.stem(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        h = x

        # Backbone
        for i, gru_layer in enumerate(self.layers):
            # GRU Forward
            h, _ = gru_layer(h)

            # Interaction (if exists for this layer)
            if i < len(self.interactions):
                h = self.interactions[i](h, pair_indices, pair_masks)

            # Dropout
            h = self.dropout(h)

        # Head
        out = self.head(h)

        return out
