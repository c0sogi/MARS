import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth (Drop Path) regularization.
    Randomly drops residual paths (samples) during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Compute shape for broadcasting: (batch_size, 1, 1, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        # Generate random tensor
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        # Scale output
        return x.div(keep_prob) * random_tensor


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation.
    Expects input of dimension 2*dim, splits it, and applies Swish gating.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split the input tensor into two halves along the last dimension
        x, gate = x.chunk(2, dim=-1)
        # Apply Swish (SiLU) to the gate and multiply
        return x * F.silu(gate)


class PreNormResBlock(nn.Module):
    """
    A Residual Block using Pre-Normalization and SwiGLU activation.
    Structure: x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))

    The Linear layer projects from dim -> 2*dim to accommodate the SwiGLU split.
    """

    def __init__(self, dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim * 2)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        input_x = x

        # Pre-Norm
        x = self.norm(x)

        # Projection (dim -> 2*dim)
        x = self.linear(x)

        # Activation (2*dim -> dim)
        x = self.swiglu(x)

        # Regularization
        x = self.dropout(x)
        x = self.drop_path(x)

        # Residual Connection
        return input_x + x


class TransitionLayer(nn.Module):
    """
    Transition layer to change dimensions between stages in the funnel.
    Structure: LayerNorm -> Linear
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x
