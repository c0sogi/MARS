import torch
import torch.nn as nn
import numpy as np
import os
from library.config import STATS_PATH, NUM_COUPLING_TYPES, RBF_RADIUS, NUM_RBF, DEVICE


class RBFExpansion(nn.Module):
    """
    Expands scalar distances into Gaussian Radial Basis Function (RBF) vectors.
    Used to create continuous filters from inter-atomic distances in the GNN backbone.
    """

    def __init__(self, vmin=0.0, vmax=RBF_RADIUS, bins=NUM_RBF):
        super(RBFExpansion, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins

        # Create centers for RBFs uniformly spaced between vmin and vmax
        # Registered as a buffer so it moves to device with the model but isn't a parameter
        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)

        # Calculate width (gamma) of the Gaussians
        # A common heuristic is gamma = 1 / step^2
        step = (vmax - vmin) / bins
        self.gamma = 1.0 / (step**2)

    def forward(self, distances):
        """
        Args:
            distances: Tensor of shape (..., ) containing scalar distances.
        Returns:
            Tensor of shape (..., bins) containing RBF expansion.
        """
        # distances: [N] -> [N, 1]
        # centers: [bins]
        # Broadcasting: [N, 1] - [bins] -> [N, bins]
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * (diff**2))


class Standardizer:
    """
    Handles per-type target standardization (Z-score normalization).
    Stores means and standard deviations for each coupling type.
    Uses pure Numpy/NPY for persistence to avoid pickle security/compatibility issues.
    """

    def __init__(self, stats_path=STATS_PATH):
        self.stats_path = stats_path
        self.means = None  # Shape (NUM_COUPLING_TYPES,)
        self.stds = None  # Shape (NUM_COUPLING_TYPES,)

        # Cache for tensor versions of stats to avoid recreation on GPU
        self.mean_t = None
        self.std_t = None

    def fit(self, types, values):
        """
        Compute mean and std for each coupling type from the training data.

        Args:
            types (np.ndarray): Array of coupling type indices (integers).
            values (np.ndarray): Array of target values.
        """
        self.means = np.zeros(NUM_COUPLING_TYPES, dtype=np.float32)
        self.stds = np.ones(NUM_COUPLING_TYPES, dtype=np.float32)

        # Ensure numpy inputs
        types = np.array(types)
        values = np.array(values)

        for i in range(NUM_COUPLING_TYPES):
            mask = types == i
            if np.any(mask):
                vals = values[mask]
                self.means[i] = np.mean(vals)
                self.stds[i] = np.std(vals)
                # Prevent division by zero
                if self.stds[i] < 1e-9:
                    self.stds[i] = 1.0

        self.save()

    def save(self):
        """Save stats to NPY file."""
        if self.means is None or self.stds is None:
            raise ValueError("Standardizer has not been fitted yet.")

        # Stack to shape (2, NUM_COUPLING_TYPES) for single file storage
        data = np.vstack([self.means, self.stds])

        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        np.save(self.stats_path, data)

    def load(self):
        """Load stats from NPY file."""
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"Stats file not found at {self.stats_path}")

        data = np.load(self.stats_path)
        self.means = data[0]
        self.stds = data[1]

    def _get_tensors(self, device):
        """Helper to get stats as tensors on the correct device."""
        if self.means is None:
            self.load()

        if self.mean_t is None or self.mean_t.device != device:
            self.mean_t = torch.tensor(self.means, dtype=torch.float32, device=device)
            self.std_t = torch.tensor(self.stds, dtype=torch.float32, device=device)

        return self.mean_t, self.std_t

    def transform(self, values, types):
        """
        Normalize values: (x - mu) / sigma
        Supports both Numpy arrays and PyTorch tensors.
        """
        if torch.is_tensor(values):
            device = values.device
            mean_t, std_t = self._get_tensors(device)
            # Gather relevant mean/std for each value based on type index
            m = mean_t[types]
            s = std_t[types]
            return (values - m) / s
        else:
            # Numpy path
            if self.means is None:
                self.load()
            m = self.means[types]
            s = self.stds[types]
            return (values - m) / s

    def inverse_transform(self, values, types):
        """
        Denormalize values: x * sigma + mu
        Supports both Numpy arrays and PyTorch tensors.
        """
        if torch.is_tensor(values):
            device = values.device
            mean_t, std_t = self._get_tensors(device)
            m = mean_t[types]
            s = std_t[types]
            return values * s + m
        else:
            # Numpy path
            if self.means is None:
                self.load()
            m = self.means[types]
            s = self.stds[types]
            return values * s + m


class MetricLogger:
    """
    Accumulates predictions and targets to compute the competition metric:
    Log of the Mean Absolute Error, averaged across coupling types.
    """

    def __init__(self):
        self.predictions = []
        self.targets = []
        self.types = []

    def reset(self):
        self.predictions = []
        self.targets = []
        self.types = []

    def update(self, preds, targets, types):
        """
        Update the logger with a batch of results.
        Args:
            preds: Tensor of predictions (unscaled/physical units).
            targets: Tensor of ground truth values.
            types: Tensor of coupling type indices.
        """
        self.predictions.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())
        self.types.append(types.detach().cpu())

    def compute_metric(self):
        """
        Compute the final metric over all accumulated batches.
        Returns:
            float: The Log Mean Absolute Error (averaged across types).
            dict: A dictionary of MAE per type for detailed analysis.
        """
        if not self.predictions:
            return 0.0, {}

        # Concatenate all batches
        all_preds = torch.cat(self.predictions).numpy()
        all_targets = torch.cat(self.targets).numpy()
        all_types = torch.cat(self.types).numpy()

        # Compute absolute errors
        abs_errors = np.abs(all_preds - all_targets)

        # Group by type and compute MAE
        mae_per_type = {}
        log_maes = []

        for type_idx in range(NUM_COUPLING_TYPES):
            mask = all_types == type_idx
            if np.any(mask):
                mae = np.mean(abs_errors[mask])
                mae_per_type[type_idx] = mae

                # Metric is Mean of Log(MAE)
                # We use natural log.
                # Safeguard against log(0) though unlikely in regression
                safe_mae = max(mae, 1e-9)
                log_maes.append(np.log(safe_mae))
            else:
                # If a type is missing from validation set, it is excluded from the average
                pass

        if not log_maes:
            return 0.0, mae_per_type

        final_metric = np.mean(log_maes)
        return final_metric, mae_per_type
