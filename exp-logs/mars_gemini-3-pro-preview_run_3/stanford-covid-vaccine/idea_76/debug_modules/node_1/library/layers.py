import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StabilizedMLP(nn.Module):
    """
    Full-Rank Stabilized MLP used for gating mechanisms.
    Includes internal LayerNorm to prevent saturation in deep networks.
    Structure: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> Sigmoid.
    """

    def __init__(self, input_dim, output_dim, hidden_ratio=2, dropout=0.1):
        super(StabilizedMLP, self).__init__()
        hidden_dim = int(input_dim * hidden_ratio)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Gating output
        )

    def forward(self, x):
        return self.net(x)


class DisentangledInteraction(nn.Module):
    """
    Topology-Disentangled Interaction Module.

    Splits processing into two distinct paths based on structural context:
    1. Paired Path (Stems): Processes interactions between base pairs (h_i, h_j).
    2. Unpaired Path (Loops): Processes self-refinement for unpaired bases (h_i).

    Uses distinct parameters for each path to avoid "Loop Silencing" and "Self-Loop Noise".
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(DisentangledInteraction, self).__init__()
        self.hidden_dim = hidden_dim

        # =====================================================================
        # Path A: Paired Interaction (Stems)
        # Source: Neighbor state h_j
        # Message: GLU(W_p1 * h_j, W_p2 * h_j)
        # Gate: MLP([h_i; h_j])
        # =====================================================================
        self.w_p1 = nn.Linear(hidden_dim, hidden_dim)
        self.w_p2 = nn.Linear(hidden_dim, hidden_dim)

        # Gate takes concatenation of self and neighbor
        self.gate_paired = StabilizedMLP(
            input_dim=hidden_dim * 2, output_dim=hidden_dim, dropout=dropout
        )

        # =====================================================================
        # Path B: Unpaired Refinement (Loops)
        # Source: Self state h_i
        # Message: GLU(W_u1 * h_i, W_u2 * h_i)
        # Gate: MLP(h_i)
        # =====================================================================
        self.w_u1 = nn.Linear(hidden_dim, hidden_dim)
        self.w_u2 = nn.Linear(hidden_dim, hidden_dim)

        # Gate takes only self
        self.gate_unpaired = StabilizedMLP(
            input_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout
        )

        # =====================================================================
        # Fusion & Normalization
        # =====================================================================
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            pair_indices: LongTensor of shape (Batch, Seq_Len)
            pair_mask: FloatTensor of shape (Batch, Seq_Len), 1.0 for paired, 0.0 for unpaired
        """
        batch_size, seq_len, _ = x.shape

        # ---------------------------------------------------------------------
        # 1. Gather Neighbor States (h_j)
        # ---------------------------------------------------------------------
        # Expand indices to match hidden dimension for gather
        # pair_indices shape: (B, L) -> (B, L, D)
        flat_indices = pair_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)

        # Gather h_j: The state of the base paired with i
        # If i is unpaired, pair_indices[i] == i, so h_j == h_i (handled by mask later)
        h_j = torch.gather(x, 1, flat_indices)

        # ---------------------------------------------------------------------
        # 2. Path A: Paired Updates
        # ---------------------------------------------------------------------
        # Message: GLU mechanism
        # content = W_p1(h_j), gate = sigmoid(W_p2(h_j))
        msg_p_content = self.w_p1(h_j)
        msg_p_gate = torch.sigmoid(self.w_p2(h_j))
        msg_paired = msg_p_content * msg_p_gate

        # Gating: How much of the message to accept based on context [h_i; h_j]
        gate_p_input = torch.cat([x, h_j], dim=-1)
        alpha_paired = self.gate_paired(gate_p_input)

        update_paired = alpha_paired * msg_paired

        # ---------------------------------------------------------------------
        # 3. Path B: Unpaired Updates
        # ---------------------------------------------------------------------
        # Message: GLU mechanism on self (h_i)
        # Uses distinct weights W_u1, W_u2
        msg_u_content = self.w_u1(x)
        msg_u_gate = torch.sigmoid(self.w_u2(x))
        msg_unpaired = msg_u_content * msg_u_gate

        # Gating: Based on self context h_i
        alpha_unpaired = self.gate_unpaired(x)

        update_unpaired = alpha_unpaired * msg_unpaired

        # ---------------------------------------------------------------------
        # 4. Fusion
        # ---------------------------------------------------------------------
        # Expand mask for broadcasting: (B, L) -> (B, L, 1)
        mask = pair_mask.unsqueeze(-1)

        # Hard switch based on topology
        # If paired (mask=1): use update_paired
        # If unpaired (mask=0): use update_unpaired
        u_total = mask * update_paired + (1.0 - mask) * update_unpaired

        # ---------------------------------------------------------------------
        # 5. Residual & Norm
        # ---------------------------------------------------------------------
        out = self.norm(x + self.dropout(u_total))

        return out
