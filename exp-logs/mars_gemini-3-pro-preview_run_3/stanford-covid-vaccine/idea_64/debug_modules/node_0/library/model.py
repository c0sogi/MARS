import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import config


class BottleneckedInteractionModule(nn.Module):
    """
    Bottlenecked Decoupled Structural Injection Module.

    Implements the strategy defined in the HC-BD-BiGRU architecture:
    1. Point-to-Point Gather of paired hidden states (h_j).
    2. Input Zero-Masking for unpaired bases (strictly avoiding self-loops).
    3. Decoupled Message Generation with Bias Refinement (m_ij).
    4. Stabilized Bottleneck MLP Gating (g_ij) with Internal Normalization.
    5. Residual Injection with Post-Normalization.
    """

    def __init__(self, hidden_dim, bottleneck_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled Message Generation
        # Projects h_j -> m_ij.
        # The bias term acts as a learnable 'loop embedding' when h_j is masked to 0.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Stabilized Bottleneck MLP Gate
        # Input: [h_i; h_j] (Dimension: 2 * hidden_dim)
        # Projects to a lower bottleneck dimension to control parameters.
        self.gate_in = nn.Linear(2 * hidden_dim, bottleneck_dim)
        self.gate_norm = nn.LayerNorm(bottleneck_dim)  # Internal Normalization
        self.gate_out = nn.Linear(bottleneck_dim, hidden_dim)

        # Post-Normalization for the residual block
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Hidden states (Batch, SeqLen, HiddenDim)
            pair_indices: Indices of paired bases (Batch, SeqLen)
            pair_mask: Mask indicating if a base is paired (1.0) or not (0.0) (Batch, SeqLen)
        """
        B, L, H = x.shape

        # 1. Gather Context (h_j)
        # Expand indices to match hidden dim: (B, L, H)
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(x, 1, idx)

        # 2. Input Zero-Masking
        # Mask unpaired contexts to 0.
        # pair_mask is float: (B, L) -> (B, L, 1)
        mask = pair_mask.unsqueeze(-1)
        h_j = h_j * mask

        # 3. Decoupled Message
        # m_ij = GELU(W * h_j + b)
        # If unpaired, h_j=0, so m_ij = GELU(b) (Bias Refinement)
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Bottlenecked Gating
        # Concatenate current state h_i and context h_j
        cat_input = torch.cat([x, h_j], dim=-1)  # (B, L, 2H)

        # Bottleneck Projection
        z_low = self.gate_in(cat_input)

        # Internal Normalization (Stabilization)
        z_norm = self.gate_norm(z_low)

        # Activation
        z_act = F.gelu(z_norm)

        # Expansion
        logits = self.gate_out(z_act)

        # Sigmoid Activation (No Logit Norm, allowing saturation)
        g_ij = torch.sigmoid(logits)

        # 5. Injection
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class HC_BD_BiGRU(nn.Module):
    """
    High-Capacity Bottlenecked-Decoupled BiGRU.

    Structure:
    - 1D Convolutional Stem
    - 4-Layer Bidirectional GRU Backbone (High Capacity: 768 total hidden dim)
    - Interleaved Bottlenecked Interaction Modules (after layers 1, 2, and 3)
    - Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.input_dim = config.INPUT_DIM
        self.stem_filters = config.STEM_FILTERS
        self.hidden_dim = config.HIDDEN_DIM  # Per direction
        self.bidirectional = config.BIDIRECTIONAL
        self.num_layers = config.NUM_LAYERS
        self.bottleneck_dim = config.BOTTLENECK_DIM
        self.num_targets = config.NUM_TARGETS
        self.dropout_rate = config.DROPOUT

        # Calculate total hidden dimension (384 * 2 = 768)
        self.total_hidden = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.stem_filters,
                kernel_size=config.STEM_KERNEL_SIZE,
                padding=config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # 2. Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        # Input dimension for the first GRU layer is the stem output
        curr_input_dim = self.stem_filters

        for i in range(self.num_layers):
            # BiGRU Layer
            gru = nn.GRU(
                input_size=curr_input_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            self.gru_layers.append(gru)

            # Interaction Module
            # Added after every GRU layer EXCEPT the final one
            if i < self.num_layers - 1:
                interaction = BottleneckedInteractionModule(
                    hidden_dim=self.total_hidden, bottleneck_dim=self.bottleneck_dim
                )
                self.interaction_layers.append(interaction)

            # Output of this GRU (and interaction) is input to next
            curr_input_dim = self.total_hidden

        # 3. Output Head
        self.head = nn.Linear(self.total_hidden, self.num_targets)

    def forward(self, sequence, pair_indices, pair_mask):
        """
        Args:
            sequence: (Batch, SeqLen, InputDim)
            pair_indices: (Batch, SeqLen)
            pair_mask: (Batch, SeqLen)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = sequence.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Backbone
        for i in range(self.num_layers):
            # GRU Layer
            # GRU returns (output, h_n), we only need output
            x, _ = self.gru_layers[i](x)

            # Interaction Module (if present for this layer)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_mask)

        # Head
        out = self.head(x)

        return out
