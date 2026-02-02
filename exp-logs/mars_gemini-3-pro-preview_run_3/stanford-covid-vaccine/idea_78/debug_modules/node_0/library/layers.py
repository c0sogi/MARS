import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """
    Convolutional Stem for Input Projection.

    Projects sparse one-hot encoded inputs into a dense embedding space and
    aggregates local k-mers using a 1D convolution.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvStem, self).__init__()
        # Padding ensures output length matches input length
        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Input_Dim)

        Returns:
            Tensor of shape (Batch, Seq_Len, Hidden_Dim)
        """
        # Conv1d expects (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        # Transpose back to (Batch, Seq_Len, Hidden_Dim)
        x = x.transpose(1, 2)
        return x


class StabilizedGLUInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    This module synthesizes Decoupled Gating, GLU Messages, and Bias-Driven Refinement.
    It gathers structural neighbor information, applies a gated linear unit (GLU)
    mechanism, and injects the information back into the sequence stream via a
    stabilized MLP gate.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(StabilizedGLUInteraction, self).__init__()
        self.hidden_dim = hidden_dim

        # =====================================================================
        # GLU Message Components
        # =====================================================================
        # W_c, b_c: Content transformation
        # If input is 0 (unpaired), output is b_c (learnable bias)
        self.w_c = nn.Linear(hidden_dim, hidden_dim)

        # W_g, b_g: Gate transformation
        # If input is 0 (unpaired), output is b_g (learnable bias)
        self.w_g = nn.Linear(hidden_dim, hidden_dim)

        # =====================================================================
        # Wide Stabilized MLP Gate Components
        # =====================================================================
        # Input is concatenation of [h_i; h_j], so input dim is 2 * hidden_dim
        # Projects to full hidden_dim (Wide Projection) to avoid bottlenecks
        self.w_in = nn.Linear(hidden_dim * 2, hidden_dim)

        # Internal Normalization to stabilize MLP internals
        self.ln_gate = nn.LayerNorm(hidden_dim)

        # Final projection for the gate scalar
        self.w_out = nn.Linear(hidden_dim, hidden_dim)

        # =====================================================================
        # Post-Processing
        # =====================================================================
        self.dropout = nn.Dropout(dropout)

        # Post-Interaction Normalization to ensure stability for the stack
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, h, adjacency, bpp_mask):
        """
        Args:
            h: Hidden states (Batch, Seq_Len, Hidden_Dim)
            adjacency: Indices of paired bases (Batch, Seq_Len)
            bpp_mask: Mask indicating paired status (Batch, Seq_Len).
                      1.0 if paired, 0.0 if unpaired.

        Returns:
            h_out: Updated hidden states (Batch, Seq_Len, Hidden_Dim)
        """
        B, L, D = h.shape

        # 1. Gather Neighbor h_j
        # Expand adjacency to match hidden dimension for gathering
        # adjacency shape: (B, L) -> (B, L, D)
        adj_expanded = adjacency.unsqueeze(-1).expand(-1, -1, D)

        # Gather h_j based on adjacency indices
        # h_j will contain the hidden state of the base paired with i
        h_j = torch.gather(h, 1, adj_expanded)

        # 2. Input Zero-Masking
        # If a base is unpaired (bpp_mask == 0), we explicitly force h_j to 0.
        # This is crucial for "Bias-Driven Refinement":
        # When h_j is 0, the Linear layers output only their bias terms.
        mask = bpp_mask.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. GLU Message (Bias-Refined)
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired bases, this becomes b_c * sigmoid(b_g)
        content = self.w_c(h_j)
        gate_signal = torch.sigmoid(self.w_g(h_j))
        m_ij = content * gate_signal

        # 4. Wide Stabilized MLP Gate
        # Calculates how much of the message m_ij should be injected

        # Concatenate current state h_i and neighbor state h_j
        concat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2*D)

        # Wide Projection
        z_raw = self.w_in(concat_input)

        # Internal Normalization and Activation
        z_norm = self.ln_gate(z_raw)
        z_act = F.gelu(z_norm)

        # Output Gate (Sigmoid to range [0, 1])
        # No Logit Norm applied here to allow full saturation
        g_ij = torch.sigmoid(self.w_out(z_act))

        # 5. Injection
        # h_struct = h + g_ij * m_ij
        update = g_ij * m_ij
        update = self.dropout(update)
        h_struct = h + update

        # 6. Post-Normalization
        h_out = self.ln_out(h_struct)

        return h_out
