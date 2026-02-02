import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledInteractionModule(nn.Module):
    """
    Implements the Stabilized Decoupled Structural Injection with Bias-Refinement.

    Logic:
    1. Gather neighbor state h_j.
    2. Mask h_j: If unpaired, h_j = 0.
    3. Message: GELU(W * h_j + b). If h_j is 0, this becomes GELU(b),
       acting as a learnable embedding for unpaired contexts (loops).
    4. Gate: Sigmoid(MLP([h_i, h_j])). Uses Internal LayerNorm for stability.
    5. Residual: h_i + gate * message.
    6. Post-Norm: LayerNorm(Residual).
    """

    def __init__(self, hidden_dim):
        super(DecoupledInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Message Projection
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating Network
        # Input is concatenation of h_i and h_j (masked)
        self.gate_proj1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_ln = nn.LayerNorm(hidden_dim)  # Internal Normalization
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization
        self.out_ln = nn.LayerNorm(hidden_dim)

    def forward(self, h, bpp_indices, bpp_masks):
        """
        Args:
            h: (Batch, Seq_Len, Hidden_Dim)
            bpp_indices: (Batch, Seq_Len) - Indices of paired bases
            bpp_masks: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, H = h.shape

        # 1. Gather h_j
        # Expand indices to match hidden dim
        # bpp_indices shape (B, L) -> (B, L, H)
        idx_expanded = bpp_indices.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(h, 1, idx_expanded)

        # 2. Zero-Masking
        # Mask shape (B, L) -> (B, L, 1)
        mask = bpp_masks.unsqueeze(-1)
        h_j_masked = h_j * mask

        # 3. Bias-Refined Message
        # If unpaired (h_j_masked is 0), output is GELU(bias)
        m_ij = F.gelu(self.msg_proj(h_j_masked))

        # 4. Stabilized MLP Gate
        # Concatenate current state and neighbor state
        cat_input = torch.cat([h, h_j_masked], dim=-1)  # (B, L, 2H)

        z_raw = self.gate_proj1(cat_input)
        z_norm = self.gate_ln(z_raw)  # Internal Normalization
        z_act = F.gelu(z_norm)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)  # No logit normalization

        # 5. Injection
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_ln(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity Stabilized Decoupled Bias-Refined BiGRU.

    Structure:
    - Conv1d Stem
    - Block 1: BiGRU + Interaction
    - Block 2: BiGRU + Interaction
    - Block 3: BiGRU + Interaction
    - Block 4: BiGRU (No Interaction)
    - Output Head
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                Config.INPUT_CHANNELS,
                Config.CONV_FILTERS,
                kernel_size=Config.CONV_KERNEL_SIZE,
                padding=Config.CONV_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # Backbone Configuration
        # BiGRU Hidden Dim is per direction, so output is 2 * HIDDEN_DIM
        self.gru_hidden_dim = Config.HIDDEN_DIM
        self.total_hidden_dim = 2 * Config.HIDDEN_DIM

        # Block 1
        self.gru1 = nn.GRU(
            Config.CONV_FILTERS,
            self.gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter1 = DecoupledInteractionModule(self.total_hidden_dim)

        # Block 2
        self.gru2 = nn.GRU(
            self.total_hidden_dim,
            self.gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter2 = DecoupledInteractionModule(self.total_hidden_dim)

        # Block 3
        self.gru3 = nn.GRU(
            self.total_hidden_dim,
            self.gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter3 = DecoupledInteractionModule(self.total_hidden_dim)

        # Block 4 (Final Block - No Interaction)
        self.gru4 = nn.GRU(
            self.total_hidden_dim,
            self.gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        self.head = nn.Linear(self.total_hidden_dim, Config.NUM_TARGETS)

    def forward(self, inputs, bpp_indices, bpp_masks):
        """
        Args:
            inputs: (B, L, 14)
            bpp_indices: (B, L)
            bpp_masks: (B, L)
        """
        # Permute for Conv1d: (B, C, L)
        x = inputs.transpose(1, 2)
        x = self.stem(x)
        x = x.transpose(1, 2)  # Back to (B, L, C)

        # Block 1
        x, _ = self.gru1(x)
        x = self.inter1(x, bpp_indices, bpp_masks)

        # Block 2
        x, _ = self.gru2(x)
        x = self.inter2(x, bpp_indices, bpp_masks)

        # Block 3
        x, _ = self.gru3(x)
        x = self.inter3(x, bpp_indices, bpp_masks)

        # Block 4
        x, _ = self.gru4(x)

        # Head
        out = self.head(x)

        return out
