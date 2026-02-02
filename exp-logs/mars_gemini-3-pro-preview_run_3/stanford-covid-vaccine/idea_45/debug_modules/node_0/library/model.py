import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionLayer(nn.Module):
    """
    Implements the Decoupled Structural Interaction Module.

    Key Components:
    - Point-to-Point Gathering via bpp_indices.
    - Input Zero-Masking for unpaired bases (Bias-Driven Refinement).
    - Decoupled Message computation.
    - Stabilized MLP Gate with Internal Normalization.
    - Residual Injection and Post-Normalization.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message generation: m_ij = GELU(W_msg * h_j + b_msg)
        self.w_msg = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism
        # Projects concatenated context [h_i; h_j]
        self.w_g1 = nn.Linear(hidden_dim * 2, hidden_dim)
        # Internal Gate Normalization (Lesson 75)
        self.ln_gate = nn.LayerNorm(hidden_dim)
        self.w_g2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization (Lesson 68)
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, x, bpp_indices, bpp_mask):
        """
        Args:
            x: Tensor of shape (B, L, D)
            bpp_indices: LongTensor of shape (B, L), indices of paired bases.
            bpp_mask: FloatTensor of shape (B, L), 1.0 if paired, 0.0 otherwise.
        """
        B, L, D = x.shape

        # 1. Gather neighbors
        # Create batch indices: (B, L)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, L)
        # Gather x_j: (B, L, D)
        x_j = x[batch_indices, bpp_indices]

        # 2. Input Zero-Masking (Lesson 64)
        # Explicitly force h_j = 0 for unpaired bases.
        # bpp_mask is (B, L) -> (B, L, 1)
        mask = bpp_mask.unsqueeze(-1)
        x_j = x_j * mask

        # 3. Decoupled Message
        # If x_j is masked to 0, this learns a bias term (Loop Embedding).
        m_ij = F.gelu(self.w_msg(x_j))

        # 4. Stabilized MLP Gate
        # Concatenate self and neighbor context
        cat_input = torch.cat([x, x_j], dim=-1)

        # Projection -> Norm -> Act -> Projection -> Sigmoid
        z_raw = self.w_g1(cat_input)
        z_norm = self.ln_gate(z_raw)  # Internal Normalization
        z_act = F.gelu(z_norm)
        logits = self.w_g2(z_act)
        g_ij = torch.sigmoid(logits)  # No Logit Norm (Lesson 78)

        # 5. Injection
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.ln_out(h_res)

        return h_out


class DSDBiGRUModel(nn.Module):
    """
    Deep Stabilized Decoupled BiGRU (DSD-BiGRU) Architecture.

    Structure:
    1. 1D Convolutional Stem.
    2. 4-Layer Backbone:
       - Layers 1-3: BiGRU -> StructuralInteractionLayer -> Dropout.
       - Layer 4: BiGRU -> LayerNorm.
    3. Linear Output Head.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.input_dim = config.INPUT_DIM
        self.conv_filters = config.CONV_FILTERS
        self.hidden_dim = config.HIDDEN_DIM
        self.num_layers = config.NUM_LAYERS
        self.dropout_prob = config.DROPOUT
        self.num_targets = config.NUM_TARGETS

        # 1. Convolutional Stem
        self.conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_filters,
            kernel_size=config.CONV_KERNEL_SIZE,
            padding=config.CONV_KERNEL_SIZE // 2,
        )
        self.act = nn.GELU()

        # 2. Deep Stabilized Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers):
            # First layer transitions from Conv filters to Hidden Dim
            # Subsequent layers stay in Hidden Dim
            in_dim = self.conv_filters if i == 0 else self.hidden_dim

            # BiGRU
            # hidden_size is half of HIDDEN_DIM because bidirectional=True doubles it
            self.gru_layers.append(
                nn.GRU(
                    input_size=in_dim,
                    hidden_size=self.hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Interaction Module logic:
            # Applied to all blocks EXCEPT the final block.
            if i < self.num_layers - 1:
                self.interaction_layers.append(
                    StructuralInteractionLayer(self.hidden_dim)
                )
                self.norms.append(None)  # Norm is inside InteractionLayer
            else:
                # Final block gets a simple LayerNorm
                self.interaction_layers.append(None)
                self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.dropout = nn.Dropout(self.dropout_prob)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, x, bpp_indices, bpp_mask):
        """
        Args:
            x: (B, L, 14) Input features.
            bpp_indices: (B, L) Structure indices.
            bpp_mask: (B, L) Structure mask.
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # Back to (B, L, C)

        for i in range(self.num_layers):
            # BiGRU
            # Output is (B, L, hidden_dim)
            x, _ = self.gru_layers[i](x)

            # Structural Interaction or Final Norm
            if self.interaction_layers[i] is not None:
                x = self.interaction_layers[i](x, bpp_indices, bpp_mask)
            else:
                x = self.norms[i](x)

            # Dropout (except potentially after the very last layer before head,
            # but standard is between blocks. Here we apply if not last block)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        # Output Head
        out = self.head(x)
        return out
