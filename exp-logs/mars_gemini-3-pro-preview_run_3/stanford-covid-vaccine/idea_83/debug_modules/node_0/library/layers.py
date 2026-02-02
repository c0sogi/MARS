import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """
    Convolutional Stem for RNA input processing.
    Projects sparse one-hot encoded inputs into a dense embedding space
    and aggregates local context via 1D convolution.
    """

    def __init__(self, input_channels=14, kernel_size=3, filters=256):
        super().__init__()
        # Padding is set to maintain sequence length (same padding)
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels)
        Returns:
            Tensor of shape (Batch, Seq_Len, Filters)
        """
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        # Permute back to (Batch, Seq_Len, Filters)
        x = x.permute(0, 2, 1)
        return x


class StabilizedGLUInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    Implements a structural injection layer that:
    1. Gathers paired hidden states (h_j) based on secondary structure.
    2. Applies a Bias-Refined GLU message mechanism.
    3. Injects structural context via a Stabilized MLP Gate with internal normalization.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GLU Message Generation: Combined Linear layer for Content (W_c) and Gate (W_g)
        # Input: h_j (masked). Output: 2 * hidden_dim (split into content and gate)
        self.glu_linear = nn.Linear(hidden_dim, hidden_dim * 2)

        # Stabilized MLP Gate
        # Input: Concatenation of [h_i; h_j] -> 2 * hidden_dim
        self.gate_w_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_ln = nn.LayerNorm(hidden_dim)  # Internal Normalization
        self.gate_act = nn.GELU()
        self.gate_w_out = nn.Linear(hidden_dim, hidden_dim)

        # Output Normalization
        self.out_ln = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_indices):
        """
        Args:
            x: Hidden states from the RNN backbone (h_i). Shape: (Batch, Seq_Len, Hidden_Dim)
            adjacency_indices: Indices of paired bases. Shape: (Batch, Seq_Len).
                               Values are indices [0, L-1]. Unpaired bases must be -1.

        Returns:
            h_out: Structurally refined hidden states. Shape: (Batch, Seq_Len, Hidden_Dim)
        """
        B, L, H = x.shape

        # 1. Create Mask and Safe Indices
        # Mask is 1.0 if paired, 0.0 if unpaired
        # Ensure mask is on the same device and dtype as x
        mask = (adjacency_indices != -1).unsqueeze(-1).type_as(x)  # (B, L, 1)

        # Replace -1 with 0 to prevent index out of bounds error during gather.
        # The values gathered at index 0 for unpaired positions will be zeroed out by the mask later.
        safe_indices = adjacency_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # 2. Gather h_j (Point-to-Point retrieval)
        # Expand indices to (B, L, H) to gather across the hidden dimension
        gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, H)
        # Gather along the sequence dimension (dim 1)
        h_j_raw = torch.gather(x, 1, gather_idx)

        # 3. Input Zero-Masking
        # If unpaired, explicitly force h_j = 0.
        # This ensures the GLU input is exactly 0 vector for unpaired bases.
        h_j = h_j_raw * mask

        # 4. GLU Message Calculation
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired bases (h_j=0), this becomes b_c * sigmoid(b_g) (Bias-Driven Refinement)
        glu_out = self.glu_linear(h_j)
        content, gate_logit = glu_out.chunk(2, dim=-1)
        m_ij = content * torch.sigmoid(gate_logit)

        # 5. Stabilized MLP Gate Calculation
        # Input: Concatenation of [h_i; h_j]
        # Note: h_i is x
        cat_input = torch.cat([x, h_j], dim=-1)

        # Wide Projection -> LayerNorm -> GELU -> Projection -> Sigmoid
        z_raw = self.gate_w_in(cat_input)
        z_norm = self.gate_ln(z_raw)
        z_act = self.gate_act(z_norm)
        g_ij = torch.sigmoid(self.gate_w_out(z_act))

        # 6. Injection and Residual Connection
        # h_struct = h_rnn + g_ij * m_ij
        injection = g_ij * m_ij
        h_struct = x + self.dropout(injection)

        # 7. Post-Normalization
        h_out = self.out_ln(h_struct)

        return h_out


class RegressionHead(nn.Module):
    """
    Lightweight Linear Regression Head for Deep Supervision.
    Predicts the 5 target values from the hidden state.
    """

    def __init__(self, input_dim, output_dim=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
