import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLUInteractionModule(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.
    Synthesizes Decoupled Gating, GLU Messages, and Bias-Driven Refinement.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GLU Message components: (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Decoupled: Only depends on h_j
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Wide Stabilized MLP Gate components
        # Input: [h_i; h_j] -> Projects to hidden_dim
        self.W_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm_gate = nn.LayerNorm(hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization
        self.layer_norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_mask):
        """
        Args:
            h: (Batch, Length, Hidden)
            pair_indices: (Batch, Length) - Indices of paired bases
            pair_mask: (Batch, Length, 1) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, H = h.shape

        # 1. Gather h_j (Point-to-Point)
        # Expand indices for gather: (B, L, H)
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(h, 1, idx)

        # 2. Input Zero-Masking
        # Explicitly force h_j = 0 for unpaired bases.
        # This ensures the linear layers W_c and W_g output only their bias terms
        # for unpaired positions, creating a learnable "loop embedding".
        h_j = h_j * pair_mask

        # 3. GLU Message (Bias-Refined)
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        msg_content = self.W_c(h_j)
        msg_gate = torch.sigmoid(self.W_g(h_j))
        m_ij = msg_content * msg_gate

        # 4. Wide Stabilized MLP Gate
        # z_raw = W_in([h_i; h_j])
        cat_input = torch.cat([h, h_j], dim=-1)
        z_raw = self.W_in(cat_input)

        # Internal Normalization & Activation
        z_norm = self.layer_norm_gate(z_raw)
        z_act = F.gelu(z_norm)

        # No Logit Norm for output, allow saturation
        g_ij = torch.sigmoid(self.W_out(z_act))

        # 5. Injection
        h_struct = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.layer_norm_out(h_struct)

        return h_out


class ResidualBiGRU(nn.Module):
    """
    Bidirectional GRU with Vertical Residual Connections.
    Allows gradients to bypass recurrent non-linearities in deep stacks.
    """

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        # hidden_size is per direction. Output size will be hidden_size * 2.
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Determine if residual connection is possible (dimensions must match)
        self.output_size = hidden_size * 2
        self.use_residual = input_size == self.output_size

    def forward(self, x):
        # x: (Batch, Length, Input_Size)
        out, _ = self.gru(x)  # out: (Batch, Length, Hidden*2)

        if self.use_residual:
            # Vertical Residual: h_l = h_{l-1} + Dropout(BiGRU(h_{l-1}))
            return x + self.dropout(out)
        else:
            # First layer (dimension change): No residual
            return out


class RNAModel(nn.Module):
    """
    Deep Residual High-Capacity BiGRU with Stabilized GLU-Interaction.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # =====================================================================
        # 1. Convolutional Stem
        # =====================================================================
        # Projects sparse inputs (14 channels) to dense embedding (256)
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.INPUT_CHANNELS,
                256,
                kernel_size=config.KERNEL_SIZE,
                padding=config.KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )
        self.stem_norm = nn.LayerNorm(256)

        # =====================================================================
        # 2. Deep Residual High-Capacity Backbone
        # =====================================================================
        self.backbone_blocks = nn.ModuleList()

        current_dim = 256
        hidden_per_dir = config.HIDDEN_DIM  # 384
        total_hidden = hidden_per_dir * 2  # 768

        for i in range(config.NUM_LAYERS):
            # Block consists of:
            # A. Vertical Residual BiGRU
            # B. Stabilized GLU-Decoupled Interaction

            # Layer 1 expands dim (256 -> 768), Layers 2-4 maintain dim (768 -> 768)
            rnn_layer = ResidualBiGRU(
                input_size=current_dim,
                hidden_size=hidden_per_dir,
                dropout=config.DROPOUT,
            )

            interaction_layer = GLUInteractionModule(total_hidden)

            self.backbone_blocks.append(
                nn.ModuleDict({"rnn": rnn_layer, "interaction": interaction_layer})
            )

            # Update dimension for next layer
            current_dim = total_hidden

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(total_hidden, 5)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: (Batch, Length, 14)
            pair_indices: (Batch, Length)
            pair_mask: (Batch, Length, 1)
        """
        # Permute for Conv1d: (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)

        # Permute back for RNN: (B, L, C)
        x = x.permute(0, 2, 1)
        h = self.stem_norm(x)

        # Pass through Deep Backbone
        for block in self.backbone_blocks:
            # A. Vertical Residual BiGRU
            h = block["rnn"](h)

            # B. Stabilized GLU-Decoupled Interaction
            h = block["interaction"](h, pair_indices, pair_mask)

        # Output Projection
        out = self.head(h)  # (B, L, 5)
        return out
