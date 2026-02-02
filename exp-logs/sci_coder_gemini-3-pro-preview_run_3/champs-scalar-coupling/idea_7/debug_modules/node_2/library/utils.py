import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands a scalar feature (distance or angle) into a vector of Radial Basis Functions (RBF).
    Used for Continuous Filter Convolutions.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        widths = torch.FloatTensor(
            np.abs(stop - start) / num_gaussians * np.ones_like(offset)
        )

        # Register as buffers so they are saved with the model and moved to device
        self.register_buffer("offset", offset)
        self.register_buffer("coeff", -0.5 / (widths**2))

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (N, 1) or (N,) containing scalar values.
        Returns:
            Tensor of shape (N, num_gaussians)
        """
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        # dist: (N, 1), offset: (num_gaussians,) -> result: (N, num_gaussians)
        diff = dist - self.offset
        return torch.exp(self.coeff * torch.pow(diff, 2))


class TargetStandardizer:
    """
    Handles per-coupling-type standardization of the target variable.
    """

    def __init__(self):
        self.stats = {}
        self.fitted = False

    def fit(self, df):
        """
        Computes mean and std for each coupling type from the training dataframe.
        Args:
            df: pandas DataFrame containing 'type' and 'scalar_coupling_constant' columns.
        """
        groups = df.groupby("type")["scalar_coupling_constant"]
        self.stats = {}

        for name, group in groups:
            mean_val = float(group.mean())
            std_val = float(group.std())
            # Avoid division by zero if std is 0 (unlikely in this dataset but good practice)
            if std_val == 0:
                std_val = 1.0

            self.stats[name] = {"mean": mean_val, "std": std_val}
        self.fitted = True
        return self

    def transform(self, df):
        """
        Standardizes the 'scalar_coupling_constant' column in the dataframe.
        z = (y - mean) / std
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before calling transform.")

        df_out = df.copy()
        # Vectorized transformation using map
        means = df_out["type"].map(lambda x: self.stats[x]["mean"])
        stds = df_out["type"].map(lambda x: self.stats[x]["std"])

        df_out["scalar_coupling_constant"] = (
            df_out["scalar_coupling_constant"] - means
        ) / stds
        return df_out

    def inverse_transform(self, pred_tensor, type_tensor):
        """
        Converts standardized predictions back to the original scale.
        Args:
            pred_tensor: torch.Tensor of shape (N,) or (N, 1) containing predicted z-scores.
            type_tensor: List of strings or numpy array of strings of length N,
                         OR torch.Tensor of integer indices mapped via Config.COUPLING_TYPE_MAP.
        Returns:
            torch.Tensor of shape (N,) containing unscaled predictions.
        """
        if not self.fitted:
            raise RuntimeError(
                "Standardizer must be fitted before calling inverse_transform."
            )

        device = pred_tensor.device

        # If type_tensor is integer indices (from DataLoader)
        if isinstance(type_tensor, torch.Tensor):
            # Create lookup tensors
            num_types = len(Config.COUPLING_TYPES)
            means_vec = torch.zeros(num_types, device=device)
            stds_vec = torch.zeros(num_types, device=device)

            for t_name, t_idx in Config.COUPLING_TYPE_MAP.items():
                if t_name in self.stats:
                    means_vec[t_idx] = self.stats[t_name]["mean"]
                    stds_vec[t_idx] = self.stats[t_name]["std"]

            # Gather specific stats for each sample
            sample_means = means_vec[type_tensor]
            sample_stds = stds_vec[type_tensor]

            return pred_tensor * sample_stds + sample_means

        # If type_tensor is list/array of strings (e.g., during manual inference)
        else:
            # Fallback to CPU/Numpy for string handling
            pred_np = pred_tensor.detach().cpu().numpy().flatten()
            types_np = np.array(type_tensor)

            result = np.zeros_like(pred_np)
            unique_types = np.unique(types_np)

            for t in unique_types:
                if t not in self.stats:
                    continue
                mask = types_np == t
                result[mask] = (
                    pred_np[mask] * self.stats[t]["std"] + self.stats[t]["mean"]
                )

            return torch.tensor(result, device=device)

    def save(self, filepath):
        """Saves the statistics to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.stats, f, indent=4)

    def load(self, filepath):
        """Loads the statistics from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stats file not found: {filepath}")

        with open(filepath, "r") as f:
            self.stats = json.load(f)
        self.fitted = True


def compute_log_mae(preds, targets, types):
    """
    Calculates the competition metric: Log of the Mean Absolute Error,
    calculated for each scalar coupling type, and then averaged across types.

    Args:
        preds: numpy array or torch tensor of predictions.
        targets: numpy array or torch tensor of ground truth.
        types: numpy array or list of coupling types (strings).

    Returns:
        float: The Log MAE score.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy().flatten()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy().flatten()

    df = pd.DataFrame({"pred": preds, "target": targets, "type": types})

    df["abs_diff"] = np.abs(df["pred"] - df["target"])

    # Calculate MAE per type
    mae_per_type = df.groupby("type")["abs_diff"].mean()

    # Take Log of MAE
    log_mae_per_type = np.log(mae_per_type)

    # Average across types
    score = log_mae_per_type.mean()

    return score


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
