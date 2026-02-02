import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledInteractionModule(nn.Module):
    """
    Implements the Decoupled Channel-Gating with Bias-Driven Loop Refinement
    and Stabilized MLP Gate.

    Logic:
    1. Gather paired hidden states h_j.
    2. Zero-mask h_j if unpaired (Input Zero-Masking).
    3. Compute message m_ij = GELU(W * h_j + b). If h_j=0, m_ij = GELU(b) (Bias-Driven Refinement).
    4. Compute gate g_ij using Stabilized MLP (Internal Norm, No Logit Norm).
    5. Inject: h_res = h_i + g_ij * m_ij.
    6. Post-Normalization.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message Generation
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

        # Stabilized MLP Gate
        # Project Joint Context: z_raw = W_g1 * [h_i; h_j]
        self.gate_proj1 = nn.Linear(2 * hidden_dim, hidden_dim)
        # Internal Normalization (stabilizes MLP internals)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        # Logit Projection
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization (stabilizes the deep backbone)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, bpp_indices, bpp_masks):
        """
        Args:
            x: (Batch, Seq, Hidden) - Input features (from BiGRU)
            bpp_indices: (Batch, Seq) - Indices of paired bases
            bpp_masks: (Batch, Seq) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, C = x.shape

        # 1. Gather h_j (Point-to-Point)
        # Expand indices to match channel dimension: (B, L, C)
        gather_idx = bpp_indices.unsqueeze(-1).expand(-1, -1, C)
        h_j = torch.gather(x, 1, gather_idx)

        # 2. Input Zero-Masking
        # Explicitly force h_j = 0 if unpaired to prevent self-loops
        mask = bpp_masks.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. Decoupled Message
        # m_ij = GELU(W * h_j + b)
        # For unpaired bases, this becomes GELU(b), serving as a loop embedding.
        m_ij = self.act(self.msg_proj(h_j))

        # 4. Stabilized MLP Gate
        # Concatenate h_i (x) and h_j
        cat_input = torch.cat([x, h_j], dim=-1)  # (B, L, 2C)

        # Project -> LayerNorm -> GELU -> Project -> Sigmoid
        z_raw = self.gate_proj1(cat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = self.act(z_norm)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)  # No Logit Norm

        # 5. Injection
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class DeepStabilizedBiGRU(nn.Module):
    """
    4-Layer Bidirectional GRU with Interleaved Decoupled Post-Norm Structural Injection.
    """

    def __init__(self):
        super().__init__()

        self.input_dim = Config.INPUT_CHANNELS
        self.conv_filters = Config.CONV_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS

        # 1. Convolutional Stem
        # Projects sparse inputs into dense embedding space
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_filters,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )
        self.stem_act = nn.GELU()

        # Projection to match Backbone Dimension (e.g. 256 -> 384)
        self.backbone_proj = nn.Linear(self.conv_filters, self.hidden_dim)

        # 2. Deep Stabilized Backbone
        self.blocks = nn.ModuleList()

        for i in range(self.num_layers):
            block = nn.ModuleDict()

            # BiGRU Layer
            # Hidden dim 384 -> Output 384.
            # Bidirectional implies hidden_size = 384 // 2 = 192 per direction.
            block["gru"] = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )

            # Structural Interaction Module (except final block)
            if i < self.num_layers - 1:
                block["interaction"] = DecoupledInteractionModule(self.hidden_dim)
            else:
                block["interaction"] = None

            self.blocks.append(block)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, 5)

    def forward(self, inputs, bpp_indices, bpp_masks):
        """
        Args:
            inputs: (B, L, 14)
            bpp_indices: (B, L)
            bpp_masks: (B, L)
        Returns:
            logits: (B, L, 5)
        """
        # Conv1d expects (B, C, L)
        x = inputs.permute(0, 2, 1)

        # Stem
        x = self.stem_conv(x)
        x = self.stem_act(x)

        # Back to (B, L, C)
        x = x.permute(0, 2, 1)

        # Project to backbone dimension
        x = self.backbone_proj(x)

        # Backbone Blocks
        for block in self.blocks:
            # BiGRU
            out, _ = block["gru"](x)

            # Update x with GRU output
            x = out

            # Interaction (if exists)
            if block["interaction"] is not None:
                x = block["interaction"](x, bpp_indices, bpp_masks)

        # Output Head
        logits = self.head(x)

        return logits
