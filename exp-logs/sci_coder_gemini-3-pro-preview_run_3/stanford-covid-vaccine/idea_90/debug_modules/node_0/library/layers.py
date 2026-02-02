import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResConv1DBlock(nn.Module):
    """
    A Residual 1D Convolutional Block.
    Structure: Conv1D -> BatchNorm -> GELU -> Residual Add.
    Maintains input dimension and sequence length.
    """

    def __init__(self, channels, kernel_size):
        super(ResConv1DBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Length).
        Returns:
            torch.Tensor: Output tensor of shape (Batch, Channels, Length).
        """
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out + residual


class DeepResidualStem(nn.Module):
    """
    Deep Residual Convolutional Stem.
    Projects raw inputs (14 channels) to the backbone hidden dimension (768).
    Hierarchy: Conv1D(k=3) -> ResBlock(k=5) -> ResBlock(k=3).
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        kernel_sizes=Config.STEM_KERNEL_SIZES,
    ):
        super(DeepResidualStem, self).__init__()

        # Ensure we have the correct number of kernels
        if len(kernel_sizes) != 3:
            # Fallback to default if mismatch
            kernel_sizes = [3, 5, 3]

        # 1. Projection Layer (Conv k=3)
        # Projects 14 -> 768
        self.proj = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            bias=False,  # Followed by BN/Act usually, but here we feed into ResBlocks
        )
        # Optional: BN/Act after projection before ResBlocks
        self.proj_bn = nn.BatchNorm1d(hidden_dim)
        self.proj_act = nn.GELU()

        # 2. Residual Block 1 (k=5)
        self.res1 = ResConv1DBlock(hidden_dim, kernel_size=kernel_sizes[1])

        # 3. Residual Block 2 (k=3)
        self.res2 = ResConv1DBlock(hidden_dim, kernel_size=kernel_sizes[2])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Length, Input_Dim).
        Returns:
            torch.Tensor: Output tensor of shape (Batch, Length, Hidden_Dim).
        """
        # Permute to (Batch, Channels, Length) for Conv1d
        x = x.permute(0, 2, 1)

        # Projection
        x = self.proj(x)
        x = self.proj_bn(x)
        x = self.proj_act(x)

        # Residual Blocks
        x = self.res1(x)
        x = self.res2(x)

        # Permute back to (Batch, Length, Channels) for RNN/Transformer backbones
        x = x.permute(0, 2, 1)
        return x


class StabilizedGLUInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.
    Synthesizes Decoupled Gating, GLU Messages, and Bias-Driven Refinement.
    """

    def __init__(self, hidden_dim=Config.HIDDEN_DIM):
        super(StabilizedGLUInteraction, self).__init__()
        self.hidden_dim = hidden_dim

        # GLU Message Components
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        self.glu_content = nn.Linear(hidden_dim, hidden_dim)
        self.glu_gate = nn.Linear(hidden_dim, hidden_dim)

        # Wide Stabilized MLP Gate
        # Input: [h_i; h_j] -> 2 * hidden_dim
        # Projects to full width (hidden_dim)
        self.gate_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_ln = nn.LayerNorm(hidden_dim)
        self.gate_act = nn.GELU()
        self.gate_out = nn.Linear(hidden_dim, hidden_dim)
        # Output activation is Sigmoid for gating

        # Post-Normalization
        self.out_ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x (torch.Tensor): Hidden states (Batch, Length, Hidden_Dim).
            pair_indices (torch.Tensor): Indices of paired bases (Batch, Length).
            pair_masks (torch.Tensor): Mask (1.0 paired, 0.0 unpaired) (Batch, Length).

        Returns:
            torch.Tensor: Refined hidden states (Batch, Length, Hidden_Dim).
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. Gather Neighbor Features (h_j)
        # Expand indices to (Batch, Length, Hidden_Dim)
        # pair_indices is (B, L). We need to gather along dim 1.
        # We need to replicate indices across the hidden dimension.
        idx_expanded = pair_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)

        # Gather h_j
        h_j = torch.gather(x, 1, idx_expanded)

        # 2. Input Zero-Masking
        # If unpaired, force h_j = 0. Strictly avoid self-loops or garbage data.
        mask_expanded = pair_masks.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask_expanded

        # 3. GLU Message (Bias-Refined)
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Note: If h_j is 0 (unpaired), this becomes b_c * sigmoid(b_g),
        # acting as a learnable bias/embedding for unpaired bases.
        content = self.glu_content(h_j)
        gate_sig = torch.sigmoid(self.glu_gate(h_j))
        m_ij = content * gate_sig

        # 4. Wide Stabilized MLP Gate
        # Concatenate h_i and h_j
        combined = torch.cat([x, h_j], dim=-1)  # (B, L, 2*H)

        # Wide Projection
        z_raw = self.gate_in(combined)

        # Internal Normalization & Activation
        z_norm = self.gate_ln(z_raw)
        z_act = self.gate_act(z_norm)

        # Gate Output (No Logit Norm, simple Sigmoid)
        g_ij = torch.sigmoid(self.gate_out(z_act))

        # 5. Injection
        # h_res = h_i + g_ij * m_ij
        h_res = x + (g_ij * m_ij)

        # 6. Post-Normalization
        h_out = self.out_ln(h_res)

        return h_out
