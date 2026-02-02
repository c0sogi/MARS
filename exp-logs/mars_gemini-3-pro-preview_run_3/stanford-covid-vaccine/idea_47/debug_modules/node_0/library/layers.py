import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """
    1D Convolutional Stem to project sparse inputs into dense embedding space.

    This layer acts as the initial feature extractor, aggregating local k-mers
    and projecting the one-hot encoded inputs into the model's hidden dimension.

    Args:
        in_channels (int): Number of input feature channels (e.g., 14).
        out_channels (int): Number of output channels (embedding dimension).
        kernel_size (int): Convolution kernel size.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvStem, self).__init__()
        # Padding is calculated to preserve sequence length: (k-1)//2 for odd k
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, L, C_in).
        Returns:
            Tensor: Output tensor of shape (N, L, C_out).
        """
        # Permute to (N, C, L) for Conv1d expectation
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.activation(x)
        # Permute back to (N, L, C) for RNN/Linear layers
        x = x.permute(0, 2, 1)
        return x


class DecoupledStructuralInteraction(nn.Module):
    """
    Decoupled Structural Interaction Module.

    This module implements the 'Stabilized Structural Evolution' strategy:
    1. Point-to-Point Gather: Retrieves hidden states of paired bases.
    2. Input Zero-Masking: Explicitly zeros out context for unpaired bases.
    3. Decoupled Message: Computes messages such that unpaired bases receive
       a learnable bias (loop embedding) instead of noise.
    4. Stabilized MLP Gate: Uses internal LayerNorm to prevent saturation.
    5. Post-Normalization: Stabilizes the residual stream for deep stacking.

    Args:
        hidden_dim (int): Hidden dimension of the input and output.
        dropout (float): Dropout probability.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(DecoupledStructuralInteraction, self).__init__()

        # Message generation: W_msg * h_j + b_msg
        # Bias is critical here as it serves as the 'loop embedding' when input is zeroed.
        self.w_msg = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Gating mechanism
        # Project Joint Context: W_g1 * [h_i; h_j]
        self.w_gate_1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

        # Internal Normalization to stabilize MLP internals (Lesson 75)
        self.ln_gate = nn.LayerNorm(hidden_dim)

        # Logit Projection
        self.w_gate_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        # Post-Normalization (Lesson 68)
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, h, adjacency):
        """
        Args:
            h (Tensor): Hidden states of shape (N, L, D).
            adjacency (Tensor): Adjacency indices of shape (N, L).
                                Values are indices of paired bases, -1 if unpaired.

        Returns:
            Tensor: Updated hidden states of shape (N, L, D).
        """
        batch_size, seq_len, hidden_dim = h.shape

        # 1. Gather Context (h_j)
        # Create a boolean mask where bases are paired (adjacency != -1)
        mask = adjacency != -1

        # Clamp indices to be valid for gather (replace -1 with 0 temporarily)
        # We use .long() for indexing
        gather_indices = adjacency.clamp(min=0).long()

        # Expand indices for gather: (N, L) -> (N, L, D)
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(
            -1, -1, hidden_dim
        )

        # Gather h_j corresponding to the paired index
        h_paired = torch.gather(h, 1, gather_indices_expanded)

        # 2. Input Zero-Masking (Lesson 64)
        # If unpaired (mask is False), explicitly force h_paired to 0.
        # This prevents self-loops or noise from index 0.
        mask_expanded = mask.unsqueeze(-1).float()
        h_paired = h_paired * mask_expanded

        # 3. Decoupled Message (Lesson 80, 85)
        # m_ij = GELU(W_msg * h_j + b)
        # For unpaired bases, h_paired is 0, so this becomes GELU(b).
        # This bias vector 'b' effectively learns a representation for loops.
        m_ij = F.gelu(self.w_msg(h_paired))

        # 4. Stabilized MLP Gate (Lesson 75, 79)
        # Concatenate h_i (current) and h_j (paired context)
        # Note: We do NOT normalize the input to preserve sparsity semantics of h_paired.
        concat_input = torch.cat([h, h_paired], dim=-1)

        # Project to hidden dim
        z_raw = self.w_gate_1(concat_input)

        # Apply Internal Normalization
        z_norm = self.ln_gate(z_raw)

        # Activation
        z_act = F.gelu(z_norm)

        # Logits and Sigmoid (No Logit Norm, Lesson 78)
        logits = self.w_gate_2(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection and Residual
        update = g_ij * m_ij
        h_res = h + self.dropout(update)

        # 6. Post-Normalization (Lesson 68)
        # Stabilizes the signal before passing to the next block
        h_out = self.ln_out(h_res)

        return h_out
