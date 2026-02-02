import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands scalar values (distances or angles) into a vector of Radial Basis Functions (RBF).
    Used for 'Stable Feature Lifting' in the Dual-Graph architecture.
    """

    def __init__(
        self, start=0.0, stop=5.0, num_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        self.centered = centered
        self.trainable = trainable

        # Compute offset (means of Gaussians) and width (beta)
        offset = torch.linspace(start, stop, num_gaussians)
        # Width is determined by the spacing between centers
        step = offset[1] - offset[0]
        beta = 1.0 / (step**2)

        if trainable:
            self.offset = nn.Parameter(offset)
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("offset", offset)
            self.register_buffer("beta", beta)

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (N, ...) containing scalar values (distances or cosines).
        Returns:
            Tensor of shape (N, ..., num_gaussians)
        """
        # (N, 1) - (num_gaussians,) -> (N, num_gaussians) via broadcasting
        diff = dist.unsqueeze(-1) - self.offset
        return torch.exp(-self.beta * torch.pow(diff, 2))


class TargetScaler:
    """
    Handles Per-Group Target Standardization.
    Calculates and applies mean/std standardization independently for each coupling type.
    """

    def __init__(self):
        self.coupling_types = Config.COUPLING_TYPES
        self.type_to_idx = {t: i for i, t in enumerate(self.coupling_types)}
        self.means = None
        self.stds = None
        self.device = Config.get_device()

    def fit(self, df, load_cached_data=True):
        """
        Computes mean and std for each coupling type from the training dataframe.
        Implements caching to avoid re-computation.

        Args:
            df (pd.DataFrame): Training metadata containing 'type' and 'scalar_coupling_constant'.
            load_cached_data (bool): If True, attempts to load stats from disk.
        """
        stats_path = Config.TARGET_STATS_PATH

        # 1. Try to load cached data
        if load_cached_data and os.path.exists(stats_path):
            try:
                stats = np.load(stats_path, allow_pickle=True).item()
                self._set_stats_from_dict(stats)
                print(f"Loaded target statistics from {stats_path}")
                return
            except Exception as e:
                print(f"Failed to load cached stats: {e}. Recomputing...")

        # 2. Compute from scratch
        print("Computing target statistics per coupling type...")
        stats = {}
        grouped = df.groupby("type")["scalar_coupling_constant"]

        for c_type in self.coupling_types:
            if c_type in grouped.groups:
                group_data = grouped.get_group(c_type)
                stats[c_type] = {
                    "mean": float(group_data.mean()),
                    "std": float(group_data.std()),
                }
            else:
                # Fallback if type not present (should not happen in full train)
                stats[c_type] = {"mean": 0.0, "std": 1.0}

        # 3. Save to cache
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        np.save(stats_path, stats)
        print(f"Saved target statistics to {stats_path}")

        self._set_stats_from_dict(stats)

    def _set_stats_from_dict(self, stats):
        """Internal helper to convert stats dict to tensors."""
        means = []
        stds = []
        for c_type in self.coupling_types:
            means.append(stats[c_type]["mean"])
            stds.append(stats[c_type]["std"])

        self.means = torch.tensor(means, dtype=torch.float32, device=self.device)
        self.stds = torch.tensor(stds, dtype=torch.float32, device=self.device)

    def transform(self, targets, types):
        """
        Standardizes targets: z = (y - mu) / sigma

        Args:
            targets (torch.Tensor): Raw target values.
            types (torch.Tensor): Integer indices of coupling types.
        """
        if self.means is None:
            raise RuntimeError("TargetScaler must be fit before transform.")

        mu = self.means[types]
        sigma = self.stds[types]
        return (targets - mu) / sigma

    def inverse_transform(self, preds, types):
        """
        Restores targets: y = z * sigma + mu

        Args:
            preds (torch.Tensor): Standardized predictions.
            types (torch.Tensor): Integer indices of coupling types.
        """
        if self.means is None:
            # If running inference without explicit fit, try loading cache
            self.fit(None, load_cached_data=True)

        mu = self.means[types]
        sigma = self.stds[types]
        return preds * sigma + mu


class AuxScaler:
    """
    Standardizes auxiliary targets (Shielding and Charges) to mean 0, std 1.
    Prevents Task Dominance in Multi-Task Learning (Cite solution_lesson_node_00009).
    """

    def __init__(self):
        self.device = Config.get_device()
        self.shield_mean = None
        self.shield_std = None
        self.charge_mean = None
        self.charge_std = None

    def fit(self, loader):
        print("Computing auxiliary target statistics...")
        s_sum = 0
        s_sq_sum = 0
        c_sum = 0
        c_sq_sum = 0
        count = 0

        # Iterate over loader to compute stats
        with torch.no_grad():
            for batch in loader:
                s = batch.y_shielding.to(self.device)
                c = batch.y_charge.to(self.device)

                s_sum += s.sum(dim=0)
                s_sq_sum += (s**2).sum(dim=0)
                c_sum += c.sum()
                c_sq_sum += (c**2).sum()
                count += s.size(0)

        self.shield_mean = s_sum / count
        self.shield_std = torch.sqrt((s_sq_sum / count) - (self.shield_mean**2) + 1e-8)

        self.charge_mean = c_sum / count
        self.charge_std = torch.sqrt((c_sq_sum / count) - (self.charge_mean**2) + 1e-8)

        print("Auxiliary stats computed.")

    def transform(self, shielding, charge):
        if self.shield_mean is None:
            raise RuntimeError("AuxScaler must be fit before transform.")

        s_std = (shielding - self.shield_mean) / self.shield_std
        c_std = (charge - self.charge_mean) / self.charge_std
        return s_std, c_std


def compute_bond_vectors(pos, edge_index):
    """
    Computes vectors and distances for edges in the graph.

    Args:
        pos (torch.Tensor): Atom positions (N, 3).
        edge_index (torch.Tensor): Graph connectivity (2, E).

    Returns:
        edge_vec (torch.Tensor): Vector from source to target (E, 3).
        edge_dist (torch.Tensor): Euclidean length of edges (E,).
    """
    row, col = edge_index
    edge_vec = pos[col] - pos[row]
    edge_dist = torch.norm(edge_vec, dim=-1)
    return edge_vec, edge_dist


def compute_bond_cosines(edge_vec, line_edge_index):
    """
    Computes the cosine of the angle between two bonds connected in the line graph.

    In the line graph, a node represents a bond (edge) in the original graph.
    An edge in the line graph connects two bonds that share an atom.

    Args:
        edge_vec (torch.Tensor): Vectors of the bonds (E_atom, 3).
        line_edge_index (torch.Tensor): Connectivity of line graph (2, E_line).
                                        Indices refer to columns in edge_vec.

    Returns:
        cosines (torch.Tensor): Cosine of angle between bond pairs (E_line,).
    """
    # line_edge_index[0] is index of bond i
    # line_edge_index[1] is index of bond j
    idx_i, idx_j = line_edge_index

    vec_i = edge_vec[idx_i]
    vec_j = edge_vec[idx_j]

    # Compute dot product
    dot = (vec_i * vec_j).sum(dim=-1)

    # Compute norms
    norm_i = torch.norm(vec_i, dim=-1)
    norm_j = torch.norm(vec_j, dim=-1)

    # Cosine = (a . b) / (|a| |b|)
    # Clamp to [-1, 1] to avoid numerical errors slightly outside range
    cosines = dot / (norm_i * norm_j + 1e-8)
    cosines = torch.clamp(cosines, -1.0, 1.0)

    return cosines
