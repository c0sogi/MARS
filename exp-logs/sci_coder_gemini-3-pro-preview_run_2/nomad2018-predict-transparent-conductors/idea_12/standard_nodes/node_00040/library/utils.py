import os
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from library.config import CUTOFF_RADIUS, MAX_NEIGHBORS


class LogStandardScaler:
    """
    Transforms target variables by applying log1p then standardization.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.device = None

    def fit(self, y):
        # y is expected to be a numpy array or torch tensor
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        y_log = np.log1p(y)
        self.mean = np.mean(y_log, axis=0)
        self.std = np.std(y_log, axis=0)
        # Handle constant features to avoid division by zero
        self.std[self.std < 1e-9] = 1.0
        return self

    def transform(self, y):
        is_tensor = False
        if isinstance(y, torch.Tensor):
            self.device = y.device
            y = y.detach().cpu().numpy()
            is_tensor = True

        y_log = np.log1p(y)
        y_scaled = (y_log - self.mean) / self.std

        if is_tensor:
            return torch.tensor(y_scaled, dtype=torch.float32, device=self.device)
        return y_scaled

    def inverse_transform(self, y_scaled):
        is_tensor = False
        if isinstance(y_scaled, torch.Tensor):
            self.device = y_scaled.device
            y_scaled = y_scaled.detach().cpu().numpy()
            is_tensor = True

        y_log = y_scaled * self.std + self.mean
        y = np.expm1(y_log)

        if is_tensor:
            return torch.tensor(y, dtype=torch.float32, device=self.device)
        return y

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


class CompositionScaler:
    """
    Standard scaler for compositional features.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.device = None

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-9] = 1.0
        return self

    def transform(self, X):
        is_tensor = False
        if isinstance(X, torch.Tensor):
            self.device = X.device
            X = X.detach().cpu().numpy()
            is_tensor = True

        X_scaled = (X - self.mean) / self.std

        if is_tensor:
            return torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        return X_scaled

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]


def build_pbc_graph(atoms: Atoms):
    """
    Converts an ASE Atoms object into a PyTorch Geometric Data object
    respecting periodic boundary conditions.
    """
    # Get atomic numbers as node features
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)

    # Calculate neighbors with PBC
    # i: source, j: target, D: distance vector
    i_indices, j_indices, S, D_vec = neighbor_list("ijSD", atoms, cutoff=CUTOFF_RADIUS)

    # Calculate scalar distances
    D = np.linalg.norm(D_vec, axis=1)

    # Convert to tensors
    edge_index = torch.tensor(np.vstack((i_indices, j_indices)), dtype=torch.long)
    edge_attr = torch.tensor(D, dtype=torch.float32).unsqueeze(1)  # Distances

    # Enforce MAX_NEIGHBORS per node
    if MAX_NEIGHBORS is not None and edge_index.shape[1] > 0:
        num_nodes = len(atoms)
        new_edge_indices_list = []
        new_edge_attrs_list = []

        for node_idx in range(num_nodes):
            # Find edges where this node is the source
            mask = edge_index[0] == node_idx

            if not mask.any():
                continue

            node_edges = edge_index[:, mask]
            node_dists = edge_attr[mask]

            if len(node_dists) > MAX_NEIGHBORS:
                # Sort by distance and keep nearest k
                # squeeze() is needed because edge_attr is (E, 1)
                sorted_indices = torch.argsort(node_dists.squeeze())
                keep_indices = sorted_indices[:MAX_NEIGHBORS]

                new_edge_indices_list.append(node_edges[:, keep_indices])
                new_edge_attrs_list.append(node_dists[keep_indices])
            else:
                new_edge_indices_list.append(node_edges)
                new_edge_attrs_list.append(node_dists)

        if new_edge_indices_list:
            edge_index = torch.cat(new_edge_indices_list, dim=1)
            edge_attr = torch.cat(new_edge_attrs_list, dim=0)
        else:
            # Handle case where filtering removes all edges (unlikely but safe)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

    data = Data(x=z, edge_index=edge_index, edge_attr=edge_attr)
    return data


def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure non-negative predictions for log
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
