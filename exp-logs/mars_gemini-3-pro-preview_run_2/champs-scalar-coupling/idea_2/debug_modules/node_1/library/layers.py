import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class ShiftedSoftplus(nn.Module):
    """
    Shifted Softplus activation function: ln(0.5 * e^x + 0.5).
    Often used in physics-based GNNs (like SchNet) for smooth, positive outputs
    that approximate linearity for x ~ 0.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        return F.softplus(x) - self.shift


def get_activation(activation_name):
    """
    Factory function to return the requested activation module.
    """
    name = activation_name.lower()
    if name == "swish" or name == "silu":
        return nn.SiLU()
    elif name == "shifted_softplus":
        return ShiftedSoftplus()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        # Default to SiLU/Swish as per config preference
        return nn.SiLU()


class RadialBasis(nn.Module):
    """
    Radial Basis Function (RBF) expansion for encoding scalar distances.
    Uses Gaussian smearing with centers linearly spaced up to the cutoff.

    Formula: exp(-beta * (d - mu_k)^2)
    """

    def __init__(self, num_rbf=Config.NUM_RBF, cutoff=Config.CUTOFF_RADIUS):
        super(RadialBasis, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # Define centers (mu_k) linearly spaced from 0 to cutoff
        # Shape: (num_rbf,)
        self.centers = nn.Parameter(
            torch.linspace(0, cutoff, num_rbf), requires_grad=False
        )

        # Determine width (beta) based on the spacing between centers
        # gap = cutoff / (num_rbf - 1)
        # We set beta = 1 / gap^2 to ensure appropriate overlap between basis functions
        if num_rbf > 1:
            gap = cutoff / (num_rbf - 1)
        else:
            gap = cutoff

        self.beta = nn.Parameter(torch.tensor(1.0 / (gap * gap)), requires_grad=False)

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of shape (..., ) containing distances.
                                      Usually (Num_Edges,).
        Returns:
            torch.Tensor: Tensor of shape (..., num_rbf) containing RBF features.
        """
        # Expand dimensions for broadcasting: (..., 1) - (1, num_rbf)
        diff = distances.unsqueeze(-1) - self.centers.view(1, -1)
        return torch.exp(-self.beta * (diff**2))


class AngularBasis(nn.Module):
    """
    Angular Basis Function (ABF) expansion for encoding triplet angles.
    Uses a Cosine Fourier series expansion.

    Formula: cos(n * theta) for n = 1...num_abf
    """

    def __init__(self, num_abf=Config.NUM_ABF):
        super(AngularBasis, self).__init__()
        self.num_abf = num_abf

        # Frequencies n = 1, 2, ..., num_abf
        self.freqs = nn.Parameter(
            torch.arange(1, num_abf + 1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, angles):
        """
        Args:
            angles (torch.Tensor): Tensor of shape (..., ) containing angles in radians.
                                   Usually (Num_Triplets,).
        Returns:
            torch.Tensor: Tensor of shape (..., num_abf) containing angular features.
        """
        # Expand dimensions for broadcasting: (..., 1) * (1, num_abf)
        return torch.cos(angles.unsqueeze(-1) * self.freqs.view(1, -1))


class DenseLayer(nn.Module):
    """
    A fundamental building block for MLPs: Linear transformation followed by an activation.
    """

    def __init__(self, in_dim, out_dim, activation=Config.ACTIVATION, bias=True):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = get_activation(activation)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization is generally robust for deep networks with Swish/SiLU
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.activation(self.linear(x))


class MLP(nn.Module):
    """
    Multi-Layer Perceptron constructed from a sequence of DenseLayers.
    Used for update functions and readout heads.
    """

    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers=2, activation=Config.ACTIVATION
    ):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(DenseLayer(in_dim, hidden_dim, activation))

        # Hidden layers
        # If num_layers=2, we have Input(Dense) -> Output(Linear)
        # If num_layers=3, we have Input(Dense) -> Hidden(Dense) -> Output(Linear)
        for _ in range(num_layers - 2):
            layers.append(DenseLayer(hidden_dim, hidden_dim, activation))

        # Output layer
        # Typically the final layer of an MLP in a GNN update block is just a Linear projection
        # to the target dimension, or a DenseLayer if further non-linearity is required.
        # Here we use a standard Linear layer for the final projection to allow flexibility.
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
