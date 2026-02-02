import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import TYPE_MAP, DEVICE


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_log_mae(preds, targets, types):
    """
    Calculates the Log Mean Absolute Error averaged across coupling types.
    Metric: Mean( Log( MAE_type ) )

    Args:
        preds (torch.Tensor): Predicted values of shape (N,).
        targets (torch.Tensor): Ground truth values of shape (N,).
        types (torch.Tensor): Coupling type indices of shape (N,).

    Returns:
        torch.Tensor: The scalar metric.
    """
    # Ensure inputs are flattened
    preds = preds.view(-1)
    targets = targets.view(-1)
    types = types.view(-1)

    unique_types = torch.unique(types)
    log_maes = []

    for t in unique_types:
        # Create a mask for the current coupling type
        mask = types == t

        if mask.sum() > 0:
            # Calculate Mean Absolute Error for this type
            mae = torch.mean(torch.abs(preds[mask] - targets[mask]))

            # Take the log of the MAE.
            # Adding a small epsilon to prevent -inf in the unlikely case of 0 error.
            log_mae = torch.log(mae + 1e-9)
            log_maes.append(log_mae)

    if not log_maes:
        return torch.tensor(0.0, device=preds.device)

    # Average the log MAEs across all present types
    return torch.stack(log_maes).mean()


class TargetScaler:
    """
    Handles per-type standardization (z-score) of the target variable.
    Scales targets independently for each coupling type.
    """

    def __init__(self):
        self.means = None
        self.stds = None
        self.device = DEVICE

    def fit(self, df):
        """
        Computes mean and std for each coupling type from the training dataframe.

        Args:
            df (pd.DataFrame): Data containing 'type' and 'scalar_coupling_constant'.
        """
        num_types = len(TYPE_MAP)
        # Initialize stats tensors
        means = torch.zeros(num_types)
        stds = torch.ones(num_types)

        for type_str, type_idx in TYPE_MAP.items():
            # Filter data for the specific coupling type
            subset = df[df["type"] == type_str]["scalar_coupling_constant"]

            if len(subset) > 0:
                mean_val = float(subset.mean())
                std_val = float(subset.std())

                # Handle edge cases where std might be 0 or NaN (e.g., single sample)
                if np.isnan(std_val) or std_val == 0:
                    std_val = 1.0

                means[type_idx] = mean_val
                stds[type_idx] = std_val

        # Move stats to the configured device
        self.means = means.to(self.device)
        self.stds = stds.to(self.device)

    def transform(self, targets, type_indices):
        """
        Standardizes targets: z = (y - mean) / std

        Args:
            targets (torch.Tensor): Raw target values.
            type_indices (torch.Tensor): Indices of coupling types corresponding to targets.

        Returns:
            torch.Tensor: Scaled targets.
        """
        if self.means is None or self.stds is None:
            raise RuntimeError("TargetScaler must be fit before calling transform.")

        # Ensure stats are on the same device as the input
        if self.means.device != targets.device:
            self.means = self.means.to(targets.device)
            self.stds = self.stds.to(targets.device)

        # Gather the mean and std for each sample based on its type
        batch_means = self.means[type_indices].view(-1, 1)
        batch_stds = self.stds[type_indices].view(-1, 1)

        return (targets - batch_means) / batch_stds

    def inverse_transform(self, scaled_targets, type_indices):
        """
        Reverts standardization: y = z * std + mean

        Args:
            scaled_targets (torch.Tensor): Scaled target values.
            type_indices (torch.Tensor): Indices of coupling types.

        Returns:
            torch.Tensor: Original scale targets.
        """
        if self.means is None or self.stds is None:
            raise RuntimeError(
                "TargetScaler must be fit before calling inverse_transform."
            )

        # Ensure stats are on the same device as the input
        if self.means.device != scaled_targets.device:
            self.means = self.means.to(scaled_targets.device)
            self.stds = self.stds.to(scaled_targets.device)

        batch_means = self.means[type_indices].view(-1, 1)
        batch_stds = self.stds[type_indices].view(-1, 1)

        return scaled_targets * batch_stds + batch_means
