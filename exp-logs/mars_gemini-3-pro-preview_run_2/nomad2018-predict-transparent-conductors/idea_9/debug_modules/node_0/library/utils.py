import torch
import numpy as np
import os
from ase import Atoms
from ase.neighborlist import neighbor_list
from library.config import Config


class GaussianSmearing(torch.nn.Module):
    """
    Expands distances into a vector of radial basis functions (Gaussian).
    """

    def __init__(self, start=0.0, stop=Config.CUTOFF_RADIUS, n_gaussians=Config.N_RBF):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        # Width is the distance between bin centers
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        self.register_buffer("offset", offset)
        self.register_buffer("width", widths)

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (N,) containing distances.
        Returns:
            Tensor of shape (N, n_gaussians) containing RBF expansion.
        """
        # (dist - offset)^2
        diff = dist.unsqueeze(-1) - self.offset
        return torch.exp(-torch.pow(diff, 2) / torch.pow(self.width, 2))


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Supports saving and loading state via .npz files for reproducibility and inference.
    """

    def __init__(self, mean=None, std=None, device=Config.DEVICE):
        self.mean = mean
        self.std = std
        self.device = device

    def fit(self, data):
        """
        Compute the mean and std to be used for later scaling.
        Args:
            data: torch.Tensor or np.ndarray
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        self.mean = torch.mean(data, dim=0).to(self.device)
        self.std = torch.std(data, dim=0).to(self.device)

        # Handle constant features (std=0) to avoid division by zero
        # Replace 0 std with 1.0 to keep values unchanged (centered at 0)
        self.std = torch.where(
            self.std == 0, torch.tensor(1.0).to(self.device), self.std
        )

    def transform(self, data):
        """
        Perform standardization by centering and scaling.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = data.to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Scale back the data to the original representation.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = data.to(self.device)
        return (data * self.std) + self.mean

    def save(self, path):
        """
        Save the mean and std to a .npz file.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy())

    def load(self, path):
        """
        Load the mean and std from a .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)


def build_pbc_graph(
    atoms: Atoms, cutoff=Config.CUTOFF_RADIUS, max_neighbors=Config.MAX_NEIGHBORS
):
    """
    Constructs a graph representation of the crystal structure considering Periodic Boundary Conditions (PBC).

    Args:
        atoms: ASE Atoms object representing the crystal structure.
        cutoff: Radius in Angstroms for neighbor search.
        max_neighbors: Maximum number of neighbors per atom (kept for compatibility,
                       though this implementation returns all neighbors within cutoff).

    Returns:
        edge_src: torch.LongTensor, source indices of edges.
        edge_dst: torch.LongTensor, destination indices of edges.
        edge_dist: torch.FloatTensor, Euclidean distances for edges.
    """
    # Use ASE's neighbor_list to find neighbors within cutoff respecting PBC
    # 'i': source index, 'j': destination index, 'd': distance
    i, j, d = neighbor_list("ijd", atoms, cutoff)

    # Convert to torch tensors
    edge_src = torch.LongTensor(i)
    edge_dst = torch.LongTensor(j)
    edge_dist = torch.FloatTensor(d)

    # Filter out self-loops (distance ~ 0) which might occur if the unit cell is small
    # and the atom sees its own periodic image at distance 0 (though rare with standard neighbor_list)
    # or if the function returns i==j with d=0.
    mask = edge_dist > 1e-6
    edge_src = edge_src[mask]
    edge_dst = edge_dst[mask]
    edge_dist = edge_dist[mask]

    return edge_src, edge_dst, edge_dist
