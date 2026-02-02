import torch
import torch.nn as nn
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Gaussian Smearing (RBF) class.
    Expands distances into a vector of Gaussian activations.
    """

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, sigma=None):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        # If sigma is not provided, estimate it from the spacing
        if sigma is None:
            sigma = (stop - start) / n_gaussians

        self.coeff = -0.5 / (sigma**2)
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: [num_edges]
        # offset: [n_gaussians]
        # Result: [num_edges, n_gaussians]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class DualScaleRBF(nn.Module):
    """
    Dual-Scale Radial Basis Function Encoder.
    Concatenates fine-grained and coarse-grained RBF expansions and projects them.
    """

    def __init__(
        self,
        fine_bins=Config.RBF_FINE_BINS,
        fine_sigma=Config.RBF_FINE_SIGMA,
        coarse_bins=Config.RBF_COARSE_BINS,
        coarse_sigma=Config.RBF_COARSE_SIGMA,
        start=Config.RBF_START,
        end=Config.RBF_END,
        hidden_dim=Config.HIDDEN_DIM,
    ):
        super(DualScaleRBF, self).__init__()

        # Fine-scale bank for precise bond lengths
        self.fine_rbf = GaussianSmearing(
            start=start, stop=end, n_gaussians=fine_bins, sigma=fine_sigma
        )

        # Coarse-scale bank for general proximity
        self.coarse_rbf = GaussianSmearing(
            start=start, stop=end, n_gaussians=coarse_bins, sigma=coarse_sigma
        )

        # Fusion Layer: Projects concatenated RBFs to hidden dimension
        input_dim = fine_bins + coarse_bins
        self.project = nn.Linear(input_dim, hidden_dim)

    def forward(self, dist):
        """
        Args:
            dist (torch.Tensor): Tensor of edge distances with shape [num_edges].
        Returns:
            torch.Tensor: Edge embeddings with shape [num_edges, hidden_dim].
        """
        fine_feat = self.fine_rbf(dist)
        coarse_feat = self.coarse_rbf(dist)

        # Concatenate features
        combined = torch.cat([fine_feat, coarse_feat], dim=-1)

        # Linear projection
        return self.project(combined)


def compute_pbc_radius_graph(
    atoms: Atoms,
    cutoff: float = Config.CUTOFF,
    max_neighbors: int = Config.MAX_NEIGHBORS,
):
    """
    Computes the Radius Graph under Periodic Boundary Conditions using ASE.

    Args:
        atoms (ase.Atoms): The atomic structure.
        cutoff (float): The cutoff radius for neighbor search.
        max_neighbors (int): Maximum number of neighbors per node to retain.

    Returns:
        dict: A dictionary containing graph tensors:
            - edge_index: [2, num_edges] LongTensor
            - edge_dist: [num_edges] FloatTensor
            - edge_vector: [num_edges, 3] FloatTensor (vectors from source to target)
            - atomic_numbers: [num_nodes] LongTensor
            - pos: [num_nodes, 3] FloatTensor
            - cell: [3, 3] FloatTensor
    """
    # Use ASE's neighbor_list to find neighbors within cutoff under PBC
    # i: source indices, j: target indices, d: distances, D: distance vectors
    i, j, d, D = neighbor_list("ijdD", atoms, cutoff)

    if len(i) == 0:
        # Handle case with no edges
        return {
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_dist": torch.empty((0,), dtype=torch.float),
            "edge_vector": torch.empty((0, 3), dtype=torch.float),
            "atomic_numbers": torch.tensor(
                atoms.get_atomic_numbers(), dtype=torch.long
            ),
            "pos": torch.tensor(atoms.get_positions(), dtype=torch.float),
            "cell": torch.tensor(atoms.get_cell().array, dtype=torch.float),
        }

    # Convert to PyTorch tensors
    edge_index = torch.stack([torch.from_numpy(i), torch.from_numpy(j)], dim=0).long()
    edge_dist = torch.from_numpy(d).float()
    edge_vector = torch.from_numpy(D).float()

    # Pruning to max_neighbors if specified
    if max_neighbors is not None and max_neighbors > 0:
        num_atoms = len(atoms)

        # Strategy: Sort edges by source node, then by distance.
        # Then select top k for each source node.

        # 1. Sort by distance (ascending)
        sort_dist_idx = torch.argsort(edge_dist)
        edge_index = edge_index[:, sort_dist_idx]
        edge_dist = edge_dist[sort_dist_idx]
        edge_vector = edge_vector[sort_dist_idx]

        # 2. Sort by source node (stable sort to preserve distance ordering within same source)
        sort_src_idx = torch.argsort(edge_index[0], stable=True)
        edge_index = edge_index[:, sort_src_idx]
        edge_dist = edge_dist[sort_src_idx]
        edge_vector = edge_vector[sort_src_idx]

        # 3. Filter to keep max_neighbors per node
        # Since we have small graphs (~80 atoms), a loop is acceptable and robust
        mask_indices = []
        for atom_idx in range(num_atoms):
            # Find indices where source == atom_idx
            # Because it's sorted, we can just find the range, but torch.where is easy
            idxs = (edge_index[0] == atom_idx).nonzero(as_tuple=True)[0]

            if len(idxs) > max_neighbors:
                # Keep only the first max_neighbors (which are the closest due to previous sort)
                mask_indices.append(idxs[:max_neighbors])
            else:
                mask_indices.append(idxs)

        if mask_indices:
            keep_mask = torch.cat(mask_indices)
            edge_index = edge_index[:, keep_mask]
            edge_dist = edge_dist[keep_mask]
            edge_vector = edge_vector[keep_mask]
        else:
            # Should not happen if i was not empty, but safe fallback
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_dist = torch.empty((0,), dtype=torch.float)
            edge_vector = torch.empty((0, 3), dtype=torch.float)

    return {
        "edge_index": edge_index,
        "edge_dist": edge_dist,
        "edge_vector": edge_vector,
        "atomic_numbers": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        "pos": torch.tensor(atoms.get_positions(), dtype=torch.float),
        "cell": torch.tensor(atoms.get_cell().array, dtype=torch.float),
    }
