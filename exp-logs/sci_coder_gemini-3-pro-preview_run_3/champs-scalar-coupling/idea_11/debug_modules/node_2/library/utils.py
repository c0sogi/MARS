import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from library.config import Config


def setup_logger(name="idea_11", log_file=None):
    """
    Sets up a logger that writes to console and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, "train.log")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def calculate_log_mae(y_true, y_pred, types):
    """
    Calculates the Log of Mean Absolute Error for each scalar coupling type,
    and then averages across types.

    Args:
        y_true (np.array or torch.Tensor): True target values.
        y_pred (np.array or torch.Tensor): Predicted values.
        types (np.array or torch.Tensor): Coupling type indices (integers).

    Returns:
        float: The competition metric score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(types, torch.Tensor):
        types = types.detach().cpu().numpy()

    # Ensure 1D arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    types = types.flatten()

    log_maes = []

    # Iterate over all defined coupling types
    # We use the mapping from Config to ensure we cover all types,
    # though we only calculate for types present in the batch/set.
    unique_types = np.unique(types)

    for t_idx in unique_types:
        mask = types == t_idx
        if np.sum(mask) == 0:
            continue

        diff = np.abs(y_true[mask] - y_pred[mask])
        mae = np.mean(diff)

        # Avoid log(0)
        if mae < 1e-9:
            mae = 1e-9

        log_mae = np.log(mae)
        log_maes.append(log_mae)

    if len(log_maes) == 0:
        return 0.0

    return np.mean(log_maes)


class Standardizer:
    """
    Handles standardization of primary targets (per coupling type) and
    auxiliary targets (global).
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.means = np.zeros(Config.NUM_COUPLING_TYPES, dtype=np.float32)
        self.stds = np.ones(Config.NUM_COUPLING_TYPES, dtype=np.float32)

        # Aux stats
        self.aux_shielding_mean = 0.0
        self.aux_shielding_std = 1.0
        self.aux_charge_mean = 0.0
        self.aux_charge_std = 1.0

        self.fitted = False

    def fit(self, df_train):
        """
        Compute mean and std for each coupling type from the training DataFrame.
        """
        print("Fitting Standardizer on training data...")
        # Ensure type column is mapped to integers if it isn't already
        if df_train["type"].dtype == "object":
            type_indices = df_train["type"].map(Config.TYPE_MAP).values
        else:
            type_indices = df_train["type"].values

        targets = df_train["scalar_coupling_constant"].values

        for t_name, t_idx in Config.TYPE_MAP.items():
            mask = type_indices == t_idx
            if np.sum(mask) > 0:
                vals = targets[mask]
                self.means[t_idx] = np.mean(vals)
                self.stds[t_idx] = np.std(vals)
                # Avoid division by zero
                if self.stds[t_idx] < 1e-6:
                    self.stds[t_idx] = 1.0

        self.fitted = True
        self.save()

    def fit_aux(self, shielding_tensor, charge_tensor):
        """
        Compute global mean and std for auxiliary targets.
        Args:
            shielding_tensor: numpy array of shape (N_atoms, 9) or flattened
            charge_tensor: numpy array of shape (N_atoms,)
        """
        self.aux_shielding_mean = np.mean(shielding_tensor)
        self.aux_shielding_std = np.std(shielding_tensor)
        if self.aux_shielding_std < 1e-6:
            self.aux_shielding_std = 1.0

        self.aux_charge_mean = np.mean(charge_tensor)
        self.aux_charge_std = np.std(charge_tensor)
        if self.aux_charge_std < 1e-6:
            self.aux_charge_std = 1.0

        self.save()

    def transform(self, df):
        """
        Apply standardization to the 'scalar_coupling_constant' column in the DataFrame.
        Returns the standardized values as a numpy array.
        """
        if not self.fitted:
            self.load()

        if "scalar_coupling_constant" not in df.columns:
            return None

        if df["type"].dtype == "object":
            type_indices = df["type"].map(Config.TYPE_MAP).values
        else:
            type_indices = df["type"].values

        targets = df["scalar_coupling_constant"].values

        # Vectorized standardization
        # Map type indices to means and stds
        mu = self.means[type_indices]
        sigma = self.stds[type_indices]

        standardized = (targets - mu) / sigma
        return standardized.astype(np.float32)

    def transform_aux(self, shielding, charge):
        """
        Standardize auxiliary targets.
        """
        if not self.fitted:
            self.load()

        s_norm = (shielding - self.aux_shielding_mean) / self.aux_shielding_std
        c_norm = (charge - self.aux_charge_mean) / self.aux_charge_std

        return s_norm.astype(np.float32), c_norm.astype(np.float32)

    def inverse_transform(self, pred_tensor, type_tensor):
        """
        Revert standardized predictions to original scale.
        Designed to work with PyTorch tensors on GPU.

        Args:
            pred_tensor: (N,) tensor of standardized predictions
            type_tensor: (N,) tensor of coupling type indices

        Returns:
            (N,) tensor of predictions in original scale
        """
        if not self.fitted:
            self.load()

        # Convert stats to tensors on the correct device
        means_t = torch.tensor(
            self.means, device=pred_tensor.device, dtype=pred_tensor.dtype
        )
        stds_t = torch.tensor(
            self.stds, device=pred_tensor.device, dtype=pred_tensor.dtype
        )

        # Gather specific mean/std for each sample
        batch_means = means_t[type_tensor]
        batch_stds = stds_t[type_tensor]

        return pred_tensor * batch_stds + batch_means

    def save(self):
        """Save stats to disk."""
        stats = {
            "means": self.means,
            "stds": self.stds,
            "aux_shielding_mean": self.aux_shielding_mean,
            "aux_shielding_std": self.aux_shielding_std,
            "aux_charge_mean": self.aux_charge_mean,
            "aux_charge_std": self.aux_charge_std,
        }
        np.save(Config.STATS_PATH, stats)
        # print(f"Standardizer stats saved to {Config.STATS_PATH}")

    def load(self):
        """Load stats from disk."""
        if os.path.exists(Config.STATS_PATH):
            stats = np.load(Config.STATS_PATH, allow_pickle=True).item()
            self.means = stats["means"]
            self.stds = stats["stds"]
            self.aux_shielding_mean = stats.get("aux_shielding_mean", 0.0)
            self.aux_shielding_std = stats.get("aux_shielding_std", 1.0)
            self.aux_charge_mean = stats.get("aux_charge_mean", 0.0)
            self.aux_charge_std = stats.get("aux_charge_std", 1.0)
            self.fitted = True
            # print(f"Standardizer stats loaded from {Config.STATS_PATH}")
        else:
            print("Warning: No stats file found. Standardizer not initialized.")
            self.fitted = False
