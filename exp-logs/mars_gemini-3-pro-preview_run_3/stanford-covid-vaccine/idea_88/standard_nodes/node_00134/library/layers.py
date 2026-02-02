import torch
import torch.nn as nn
import torch.nn.functional as F


class StabilizedGLUInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Structural Injection Module.

    Implements the structural interaction logic defined in the High-Capacity Stabilized Strategy:
    1. Gathers neighbor features using an adjacency map.
    2. Applies zero-masking for unpaired bases (h_j = 0).
    3. Computes Decoupled GLU message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
       For unpaired bases, this becomes a learnable bias term (Bias-Driven Refinement).
    4. Computes Wide Stabilized MLP Gate:
       z = LayerNorm(W_in([h_i, h_j])) -> GELU -> Sigmoid(W_out(z))
    5. Performs residual injection with Post-LayerNorm.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        """
        Args:
            hidden_dim (int): The hidden dimension of the backbone (e.g., 768).
            dropout (float): Dropout probability.
        """
        super(StabilizedGLUInteraction, self).__init__()
        self.hidden_dim = hidden_dim

        # =====================================================================
        # GLU Message Components (Decoupled)
        # =====================================================================
        # Operates purely on h_j.
        # If h_j is masked (0), these act as learnable biases (b_c * sigmoid(b_g)).
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # =====================================================================
        # Wide Stabilized MLP Gate Components
        # =====================================================================
        # Input is concatenation of h_i and h_j -> 2 * hidden_dim
        # Projects to full width (hidden_dim) to avoid bottlenecks.
        self.W_in = nn.Linear(2 * hidden_dim, hidden_dim)

        # Internal Normalization to stabilize MLP internals
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # Output projection for the gate
        self.W_out = nn.Linear(hidden_dim, hidden_dim)

        # =====================================================================
        # Post-Processing
        # =====================================================================
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Hidden_Dim). Represents h_i.
            pair_indices: LongTensor of shape (Batch, Seq_Len). Contains indices of paired bases.
                          Indices must be valid (0 to Seq_Len-1). Unpaired positions can point
                          to self or any valid index, as they will be masked out.
            pair_mask: Tensor of shape (Batch, Seq_Len) or (Batch, Seq_Len, 1).
                       1.0 if paired, 0.0 if unpaired.

        Returns:
            Tensor of shape (Batch, Seq_Len, Hidden_Dim).
        """
        batch_size, seq_len, _ = x.size()

        # ---------------------------------------------------------------------
        # 1. Gather & Mask (Structural Context)
        # ---------------------------------------------------------------------
        # Expand pair_indices to match hidden dimension for gathering
        # Shape: (B, L) -> (B, L, D)
        idx_expanded = pair_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)

        # Gather h_j (neighbor features)
        # x is (B, L, D). We gather along dim 1 (sequence length).
        h_j = torch.gather(x, 1, idx_expanded)

        # Input Zero-Masking
        # Ensure mask is broadcastable: (B, L) -> (B, L, 1)
        if pair_mask.dim() == 2:
            pair_mask = pair_mask.unsqueeze(-1)

        # Explicitly force h_j = 0 if unpaired.
        h_j = h_j * pair_mask

        # ---------------------------------------------------------------------
        # 2. Decoupled GLU Message (Bias-Refined)
        # ---------------------------------------------------------------------
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        content = self.W_c(h_j)
        gate_msg = torch.sigmoid(self.W_g(h_j))
        m_ij = content * gate_msg

        # ---------------------------------------------------------------------
        # 3. Wide Stabilized MLP Gate
        # ---------------------------------------------------------------------
        # Concatenate h_i and h_j
        cat_input = torch.cat([x, h_j], dim=-1)  # (B, L, 2*D)

        # Wide Projection -> LayerNorm -> GELU
        z_raw = self.W_in(cat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)

        # Sigmoid output for gating (No Logit Norm here to allow saturation)
        g_ij = torch.sigmoid(self.W_out(z_act))

        # ---------------------------------------------------------------------
        # 4. Injection & Post-Normalization
        # ---------------------------------------------------------------------
        # h_res = h_i + g_ij * m_ij
        update = g_ij * m_ij
        update = self.dropout(update)

        h_res = x + update

        # Final stabilization
        h_out = self.out_norm(h_res)

        return h_out
