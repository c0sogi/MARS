import torch
import torch.nn as nn
import torch.nn.functional as F


class StabilizedGLUInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    This module synthesizes Decoupled Gating, GLU Messages, and Bias-Driven Refinement
    to inject structural information into the sequence representation.

    Mechanism:
    1. Gather: Retrieves neighbor features h_j based on adjacency.
    2. Zero-Masking: Forces h_j=0 for unpaired bases, allowing bias terms to learn loop embeddings.
    3. GLU Message: Computes m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
    4. Wide Gate: Computes gating coefficient g_ij via a wide, internally normalized MLP on [h_i; h_j].
    5. Injection: Updates stream via h_out = Norm(h_i + g_ij * m_ij).
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------------------
        # GLU Message Components (Decoupled)
        # ----------------------------------------------------------------
        # Message Content: W_c * h_j + b_c
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        # Message Gate: sigmoid(W_g * h_j + b_g)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------------------
        # Wide Stabilized MLP Gate
        # ----------------------------------------------------------------
        # Projects concatenation of [h_i; h_j] to full width (hidden_dim)
        # Input dim: 2 * hidden_dim
        self.gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        # Internal Normalization for stability
        self.gate_norm = nn.LayerNorm(hidden_dim)
        # Output Projection (No Logit Norm to allow full saturation)
        self.gate_out = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------------------
        # Regularization & Output
        # ----------------------------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adjacency, pair_mask):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Hidden_Dim).
            adjacency (torch.Tensor): Adjacency indices (Batch, Seq_Len).
                                      Unpaired bases should be -1.
            pair_mask (torch.Tensor): Mask for paired bases (Batch, Seq_Len).
                                      1.0 if paired, 0.0 if unpaired.

        Returns:
            torch.Tensor: Updated features (Batch, Seq_Len, Hidden_Dim).
        """
        B, L, C = x.shape

        # ----------------------------------------------------------------
        # 1. Gather Neighbor Features
        # ----------------------------------------------------------------
        # Handle sentinel -1 in adjacency. Clamp to 0 to avoid index errors.
        # The values gathered from index 0 for unpaired bases will be masked out in step 2.
        safe_adj = adjacency.clone()
        safe_adj[safe_adj < 0] = 0

        # Expand indices for gather: (B, L) -> (B, L, C)
        gather_idx = safe_adj.unsqueeze(-1).expand(-1, -1, C)

        # Gather h_j: (B, L, C)
        h_j = torch.gather(x, 1, gather_idx)

        # ----------------------------------------------------------------
        # 2. Input Zero-Masking
        # ----------------------------------------------------------------
        # Explicitly force h_j = 0 for unpaired bases.
        # This enables the bias terms in the GLU message to act as a learned "unpaired embedding".
        # pair_mask is (B, L), expand to (B, L, 1)
        mask = pair_mask.unsqueeze(-1)
        h_j = h_j * mask

        # ----------------------------------------------------------------
        # 3. GLU Message (Bias-Refined)
        # ----------------------------------------------------------------
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Note: For unpaired bases (h_j=0), this becomes b_c * sigmoid(b_g).
        content = self.W_c(h_j)
        gate_signal = torch.sigmoid(self.W_g(h_j))
        m_ij = content * gate_signal

        # ----------------------------------------------------------------
        # 4. Wide Stabilized MLP Gate
        # ----------------------------------------------------------------
        # Concatenate [h_i; h_j]
        concat_input = torch.cat([x, h_j], dim=-1)

        # Wide Projection -> LayerNorm -> GELU
        z_raw = self.gate_proj(concat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)

        # Final Gate coefficient: g_ij = sigmoid(W_out * z_act)
        g_ij = torch.sigmoid(self.gate_out(z_act))

        # ----------------------------------------------------------------
        # 5. Injection & Post-Normalization
        # ----------------------------------------------------------------
        # h_struct = h_i + g_ij * m_ij
        update = g_ij * m_ij
        update = self.dropout(update)

        h_struct = x + update
        h_out = self.out_norm(h_struct)

        return h_out


class VerticalResBiGRU(nn.Module):
    """
    Bidirectional GRU with Vertical Residual Connections.

    Allows gradients to bypass recurrent non-linearities, facilitating the training
    of deep backbones (e.g., 4+ layers).
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        """
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Total dimension of output features (forward + backward).
            dropout (float): Dropout probability.
        """
        super().__init__()

        # BiGRU: hidden_size per direction is half of the total hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Determine if residual connection is valid (dimensions must match)
        self.use_residual = input_dim == hidden_dim

    def forward(self, x):
        # x: (B, L, input_dim)
        # out: (B, L, hidden_dim)
        out, _ = self.gru(x)

        if self.use_residual:
            # Vertical Residual: h_l = h_{l-1} + BiGRU(h_{l-1})
            out = x + self.dropout(out)
        else:
            # Just dropout if dimensions changed (e.g., first layer after stem)
            out = self.dropout(out)

        return out
