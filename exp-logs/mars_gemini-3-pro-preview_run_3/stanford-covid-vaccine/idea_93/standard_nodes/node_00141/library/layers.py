import torch
import torch.nn as nn
import torch.nn.functional as F


class StabilizedGLUInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    This module synthesizes structural information by:
    1. Gathering neighbor states (h_j) based on pairing.
    2. Applying a decoupled GLU message function (allows bias-driven loop embeddings).
    3. Fusing information via a wide, stabilized MLP gate.
    """

    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim (int): The feature dimension of the input tensor (e.g., 768).
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------------------
        # 1. GLU Message Components (Decoupled)
        # Message m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Standard nn.Linear includes bias, which is crucial for unpaired bases.
        # ----------------------------------------------------------------
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------------------
        # 2. Wide Stabilized MLP Gate
        # Input: Concat[h_i, h_j] -> 2 * hidden_dim
        # Projects to full width (hidden_dim) to avoid information bottlenecks.
        # Includes LayerNorm before activation for stability.
        # ----------------------------------------------------------------
        self.gate_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_out = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------------------
        # 3. Post-Injection Normalization
        # ----------------------------------------------------------------
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj, mask):
        """
        Args:
            h (torch.Tensor): Input sequence features. Shape (Batch, Length, Hidden).
            adj (torch.Tensor): Adjacency indices for pairing. Shape (Batch, Length).
            mask (torch.Tensor): Pairing mask (1.0 for paired, 0.0 for unpaired). Shape (Batch, Length).

        Returns:
            torch.Tensor: Refined sequence features. Shape (Batch, Length, Hidden).
        """
        B, L, H = h.shape

        # ----------------------------------------------------------------
        # Step 1: Gather Neighbor States (h_j)
        # ----------------------------------------------------------------
        # Expand adj to cover the feature dimension: (B, L, H)
        adj_expanded = adj.unsqueeze(-1).expand(-1, -1, H)

        # Gather h_j. Note: adj contains 0 for unpaired bases (masked later),
        # so we gather index 0 temporarily for them.
        h_j = torch.gather(h, 1, adj_expanded)

        # ----------------------------------------------------------------
        # Step 2: Input Zero-Masking
        # ----------------------------------------------------------------
        # Force h_j = 0 for unpaired bases.
        # This ensures the GLU message relies purely on the bias terms for loops.
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask_expanded

        # ----------------------------------------------------------------
        # Step 3: GLU Message (Bias-Refined)
        # ----------------------------------------------------------------
        # m_ij = Content * Gate
        # For unpaired bases (h_j=0), this becomes: bias_c * sigmoid(bias_g)
        msg_content = self.W_c(h_j)
        msg_gate = torch.sigmoid(self.W_g(h_j))
        m_ij = msg_content * msg_gate

        # ----------------------------------------------------------------
        # Step 4: Wide Stabilized MLP Gate
        # ----------------------------------------------------------------
        # g_ij = sigmoid(MLP([h_i, h_j]))
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2H)

        # Project
        z_raw = self.gate_in(cat_input)  # (B, L, H)

        # Stabilize with Norm + Act
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)

        # Compute Gate
        g_ij = torch.sigmoid(self.gate_out(z_act))

        # ----------------------------------------------------------------
        # Step 5: Injection and Normalization
        # ----------------------------------------------------------------
        # Residual connection weighted by the learned gate
        h_res = h + g_ij * m_ij

        # Final LayerNorm
        h_out = self.out_norm(h_res)

        return h_out
