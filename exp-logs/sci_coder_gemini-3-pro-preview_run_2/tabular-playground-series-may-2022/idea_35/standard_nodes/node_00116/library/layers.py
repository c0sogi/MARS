import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


class LayerScale(nn.Module):
    """
    LayerScale layer with a learnable diagonal matrix.
    Scales the output of a block by a learnable factor per channel.
    Initialized to a small value (e.g., 1e-5) to improve training stability in deep networks.
    """

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.
    Expects input of size (..., 2 * dim).
    Splits input into two halves: val and gate.
    Computes: val * SiLU(gate).
    Output size is (..., dim).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class LayerScaledBlock(nn.Module):
    """
    LayerScaled Pre-Norm SwiGLU Residual Block.
    Structure: x + DropPath(LayerScale(Dropout(SwiGLU(Linear(LayerNorm(x))))))

    This block uses a Gated Linear Unit design where a single Linear layer
    expands the dimension to 2x, and the SwiGLU activation reduces it back to 1x
    via gating, acting as both the non-linearity and the projection.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-5, dropout=0.0):
        super().__init__()

        # Pre-Normalization
        self.norm = nn.LayerNorm(dim)

        # Linear Projection (dim -> 2*dim for Gating)
        self.linear = nn.Linear(dim, 2 * dim)

        # Activation (2*dim -> dim)
        self.act = SwiGLU()

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Scaling
        self.ls = LayerScale(dim, init_values=layer_scale_init)

        # Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.ls(x)
        x = self.drop_path(x)

        return x + shortcut
