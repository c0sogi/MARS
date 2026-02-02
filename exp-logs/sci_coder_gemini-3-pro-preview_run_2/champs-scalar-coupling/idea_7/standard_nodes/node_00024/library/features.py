import torch
import torch.nn as nn
from library.config import Config

# Try to import from torch_geometric, which is installed in the environment
try:
    from torch_geometric.nn.models.dimenet import BesselBasisLayer, SphericalBasisLayer
except ImportError:
    raise ImportError(
        "torch_geometric is required for RBF and SBF expansions but could not be imported."
    )


class RBFExpansion(nn.Module):
    """
    Expands inter-atomic distances using Bessel Basis Functions.
    This serves as the geometric encoding for edge features in the graph.
    """

    def __init__(
        self,
        num_rbf: int = Config.NUM_RBF,
        cutoff: float = Config.CUTOFF,
        envelope_exponent: int = 5,
    ):
        """
        Args:
            num_rbf (int): Number of radial basis functions.
            cutoff (float): Cutoff distance in Angstroms.
            envelope_exponent (int): Exponent for the polynomial envelope function.
        """
        super(RBFExpansion, self).__init__()
        # BesselBasisLayer implements the formula: sqrt(2/c) * sin(n*pi*d/c) / d * envelope(d)
        self.rbf_layer = BesselBasisLayer(
            num_radial=num_rbf, cutoff=cutoff, envelope_exponent=envelope_exponent
        )

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist (torch.Tensor): Tensor of shape (num_edges,) containing Euclidean distances.

        Returns:
            torch.Tensor: Tensor of shape (num_edges, num_rbf) containing expanded features.
        """
        return self.rbf_layer(dist)


class SBFExpansion(nn.Module):
    """
    Expands triplet bond angles using Spherical Basis Functions (Spherical Harmonics + Radial Basis).
    This encodes the angular information for triplets (atom i, atom j, atom k).
    """

    def __init__(
        self,
        num_sbf: int = Config.NUM_SBF,
        num_rbf: int = Config.NUM_RBF,
        cutoff: float = Config.CUTOFF,
        envelope_exponent: int = 5,
    ):
        """
        Args:
            num_sbf (int): Number of spherical basis functions.
            num_rbf (int): Number of radial basis functions.
            cutoff (float): Cutoff distance in Angstroms.
            envelope_exponent (int): Exponent for the polynomial envelope function.
        """
        super(SBFExpansion, self).__init__()
        # SphericalBasisLayer implements the 2D spherical Fourier-Bessel basis
        self.sbf_layer = SphericalBasisLayer(
            num_spherical=num_sbf,
            num_radial=num_rbf,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )

    def forward(
        self, dist: torch.Tensor, angle: torch.Tensor, idx_kj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dist (torch.Tensor): Tensor of shape (num_edges,) containing distances for all edges.
            angle (torch.Tensor): Tensor of shape (num_triplets,) containing bond angles in radians.
            idx_kj (torch.Tensor): Tensor of shape (num_triplets,) containing the edge indices
                                   corresponding to the 'kj' leg of the triplet. This is used to
                                   gather the correct radial features from 'dist'.

        Returns:
            torch.Tensor: Tensor of shape (num_triplets, num_sbf * num_rbf) containing expanded features.
        """
        return self.sbf_layer(dist, angle, idx_kj)
