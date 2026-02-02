import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.
    Expects the input tensor to have a size of 2 * hidden_dim in the last dimension.
    It splits the input into two halves (a, b), applies Swish (SiLU) to a,
    and returns swish(a) * b.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split the input into two equal halves along the last dimension
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
