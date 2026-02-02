import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Expands scalar values (distances or angles) into a vector of Gaussian Radial Basis Functions (RBF).

    This is used for Continuous Filter Convolutions (CFConv) to allow the network to learn
    non-linear representations of geometry.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        # Create a tensor of centers for the Gaussians
        offset = torch.linspace(start, stop, num_gaussians)

        # Calculate the width (gamma) of the Gaussians based on the spacing
        # coeff = -1 / (2 * sigma^2)
        # We set sigma to be the distance between centers
        step = offset[1] - offset[0]
        self.coeff = -0.5 / (step.item() ** 2)

        # Register offset as a buffer so it's part of the state_dict but not a trainable parameter
        self.register_buffer("offset", offset)

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape (N,) or (N, 1) containing scalar values.

        Returns:
            Tensor of shape (N, num_gaussians)
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TargetScaler:
    """
    Handles per-group standardization of the target variable (scalar_coupling_constant).

    Also provides utilities for standardizing auxiliary targets.
    """

    def __init__(self, device=None):
        self.means = {}
        self.stds = {}
        self.device = device if device else torch.device("cpu")

        # Tensors for vectorized operations (populated after fit or load)
        self.mean_tensor = None
        self.std_tensor = None

        # Auxiliary stats
        self.aux_stats = {}

    def fit(self, df):
        """
        Computes mean and std for each coupling type from the training dataframe.

        Args:
            df: Pandas DataFrame containing 'type' and 'scalar_coupling_constant'.
        """
        print("Fitting TargetScaler on training data...")
        groups = df.groupby("type")["scalar_coupling_constant"]

        for coupling_type, group in groups:
            self.means[coupling_type] = group.mean()
            self.stds[coupling_type] = group.std()

        # Prepare tensors for vectorized lookup based on Config.COUPLING_TYPES indices
        num_types = len(Config.COUPLING_TYPES)
        mean_list = []
        std_list = []

        for i in range(num_types):
            c_type = Config.COUPLING_TYPES[i]
            if c_type in self.means:
                mean_list.append(self.means[c_type])
                std_list.append(self.stds[c_type])
            else:
                # Fallback if a type is missing in training (unlikely)
                mean_list.append(0.0)
                std_list.append(1.0)

        self.mean_tensor = torch.tensor(
            mean_list, dtype=torch.float32, device=self.device
        )
        self.std_tensor = torch.tensor(
            std_list, dtype=torch.float32, device=self.device
        )
        print("TargetScaler fit complete.")

    def fit_auxiliary(self, shielding_vals, charge_vals):
        """
        Computes global mean/std for auxiliary targets.

        Args:
            shielding_vals: Numpy array or Tensor of magnetic shielding values.
            charge_vals: Numpy array or Tensor of mulliken charge values.
        """
        self.aux_stats["shielding_mean"] = float(np.mean(shielding_vals))
        self.aux_stats["shielding_std"] = float(np.std(shielding_vals))
        self.aux_stats["charge_mean"] = float(np.mean(charge_vals))
        self.aux_stats["charge_std"] = float(np.std(charge_vals))

    def transform(self, targets, type_indices):
        """
        Standardizes targets based on their coupling type.
        z = (y - mean_type) / std_type

        Args:
            targets: Tensor of shape (N,) containing raw target values.
            type_indices: Tensor of shape (N,) containing integer indices of coupling types.

        Returns:
            Tensor of shape (N,) containing standardized values.
        """
        if self.mean_tensor is None:
            raise RuntimeError(
                "TargetScaler must be fitted or loaded before transform."
            )

        # Ensure tensors are on the correct device
        if targets.device != self.device:
            targets = targets.to(self.device)
        if type_indices.device != self.device:
            type_indices = type_indices.to(self.device)

        means = self.mean_tensor[type_indices]
        stds = self.std_tensor[type_indices]

        return (targets - means) / stds

    def inverse_transform(self, preds, type_indices):
        """
        Reverts standardized predictions to physical scale.
        y = z * std_type + mean_type

        Args:
            preds: Tensor of shape (N,) containing standardized predictions.
            type_indices: Tensor of shape (N,) containing integer indices of coupling types.

        Returns:
            Tensor of shape (N,) containing physical values.
        """
        if self.mean_tensor is None:
            raise RuntimeError(
                "TargetScaler must be fitted or loaded before inverse_transform."
            )

        # Ensure tensors are on the correct device
        if preds.device != self.device:
            preds = preds.to(self.device)
        if type_indices.device != self.device:
            type_indices = type_indices.to(self.device)

        means = self.mean_tensor[type_indices]
        stds = self.std_tensor[type_indices]

        return preds * stds + means

    def transform_auxiliary(self, shielding, charges):
        """
        Standardizes auxiliary targets globally.

        Args:
            shielding: Tensor of shielding values.
            charges: Tensor of charge values.

        Returns:
            Tuple of (standardized_shielding, standardized_charges)
        """
        if "shielding_mean" not in self.aux_stats:
            # If not fitted, return as is (or raise error, but safe fallback is identity)
            return shielding, charges

        s_mean = self.aux_stats["shielding_mean"]
        s_std = self.aux_stats["shielding_std"]
        c_mean = self.aux_stats["charge_mean"]
        c_std = self.aux_stats["charge_std"]

        s_norm = (shielding - s_mean) / s_std
        c_norm = (charges - c_mean) / c_std

        return s_norm, c_norm

    def state_dict(self):
        """Returns the state dictionary for saving."""
        return {
            "means": self.means,
            "stds": self.stds,
            "mean_tensor": (
                self.mean_tensor.cpu() if self.mean_tensor is not None else None
            ),
            "std_tensor": (
                self.std_tensor.cpu() if self.std_tensor is not None else None
            ),
            "aux_stats": self.aux_stats,
        }

    def load_state_dict(self, state_dict):
        """Loads the state dictionary."""
        self.means = state_dict["means"]
        self.stds = state_dict["stds"]
        self.aux_stats = state_dict.get("aux_stats", {})

        if state_dict["mean_tensor"] is not None:
            self.mean_tensor = state_dict["mean_tensor"].to(self.device)
        if state_dict["std_tensor"] is not None:
            self.std_tensor = state_dict["std_tensor"].to(self.device)

    def save(self, path):
        """Saves the scaler state to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Loads the scaler state from disk."""
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
        else:
            raise FileNotFoundError(f"Scaler state file not found at {path}")
