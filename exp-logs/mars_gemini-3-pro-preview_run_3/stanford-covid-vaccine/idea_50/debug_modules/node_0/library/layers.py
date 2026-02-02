import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module (SDBR-BiGRU Strategy).

    This module implements a stabilized interaction mechanism that:
    1. Gathers paired hidden states (h_j).
    2. Applies Input Zero-Masking to strictly separate paired/unpaired dynamics.
    3. Computes a Decoupled Message where unpaired bases rely on learned bias terms.
    4. Uses a Stabilized MLP Gate with internal LayerNorm for robust optimization.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled Message Projection
        # Logic: m_ij = GELU(W_msg * h_j + b_msg)
        # For unpaired bases (h_j=0), this learns a bias embedding.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stabilized MLP Gate components
        # 1. Projection of Joint Context [h_i; h_j]
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)

        # 2. Internal Normalization (Key for stability)
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # 3. Logit Projection
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization for the residual connection
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim). Represents h_i.
            pair_indices: Tensor of shape (Batch, Seq_Len). Indices of paired bases.
            pair_mask: Tensor of shape (Batch, Seq_Len). 1.0 if paired, 0.0 if unpaired.

        Returns:
            Tensor of shape (Batch, Seq_Len, Hidden_Dim).
        """
        B, L, D = x.shape

        # -------------------------------------------------------
        # 1. Gather h_j (Point-to-Point Interaction)
        # -------------------------------------------------------
        # Flatten x to (B*L, D) to allow indexing with flat_indices
        x_flat = x.view(-1, D)

        # Calculate flat indices: batch_offset + pair_index
        # pair_indices values are in [0, L-1]
        batch_offsets = torch.arange(B, device=x.device).unsqueeze(1) * L
        flat_indices = (batch_offsets + pair_indices).view(-1)

        # Gather and reshape back to (B, L, D)
        h_j_raw = x_flat[flat_indices].view(B, L, D)

        # -------------------------------------------------------
        # 2. Input Zero-Masking
        # -------------------------------------------------------
        # If unpaired (mask=0), force h_j = 0.
        # This prevents self-loops or arbitrary index noise and enables
        # the bias-driven refinement for unpaired bases.
        mask = pair_mask.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j_raw * mask

        # -------------------------------------------------------
        # 3. Decoupled Message
        # -------------------------------------------------------
        # m_ij = GELU(W_msg * h_j + b_msg)
        # We do NOT concatenate h_i here.
        m_ij = F.gelu(self.msg_proj(h_j))

        # -------------------------------------------------------
        # 4. Stabilized MLP Gate
        # -------------------------------------------------------
        # Joint Context: [h_i; h_j]
        cat_input = torch.cat([x, h_j], dim=-1)

        # Project -> Internal Norm -> Act -> Project -> Sigmoid
        z_raw = self.gate_proj1(cat_input)
        z_norm = self.gate_norm(z_raw)  # Internal Normalization
        z_act = F.gelu(z_norm)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)  # No logit norm

        # -------------------------------------------------------
        # 5. Injection & Post-Normalization
        # -------------------------------------------------------
        # h_res = h_i + g_ij * m_ij
        update = g_ij * m_ij
        h_res = x + self.dropout(update)

        # Stabilize output for the next recurrent layer
        h_out = self.out_norm(h_res)

        return h_out
