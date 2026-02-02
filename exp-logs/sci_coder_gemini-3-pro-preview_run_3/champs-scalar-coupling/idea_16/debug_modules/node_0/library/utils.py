import os
import torch
import numpy as np
import pandas as pd
from library.config import Config


class Standardizer:
    """
    Handles per-coupling-type standardization of target values.
    Stores means and standard deviations for each of the 8 coupling types.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.means = None
        self.stds = None
        self.num_types = Config.NUM_COUPLING_TYPES
        self.stats_path = Config.STATS_PATH

    def fit(self, df=None, load_cached_data=True):
        """
        Computes mean and std per coupling type from the training dataframe.

        Args:
            df (pd.DataFrame, optional): Training metadata containing 'type' and 'scalar_coupling_constant'.
                                         Required if stats are not loaded from cache.
            load_cached_data (bool): If True, attempts to load stats from Config.STATS_PATH.
        """
        if load_cached_data and os.path.exists(self.stats_path):
            print(f"Loading target statistics from {self.stats_path}...")
            stats = np.load(self.stats_path, allow_pickle=True).item()
            self.means = torch.tensor(
                stats["means"], dtype=torch.float32, device=self.device
            )
            self.stds = torch.tensor(
                stats["stds"], dtype=torch.float32, device=self.device
            )
            return

        if df is None:
            raise ValueError(
                "DataFrame must be provided if cached stats are not available or load_cached_data=False."
            )

        print("Computing target statistics from training data...")
        means = np.zeros(self.num_types, dtype=np.float32)
        stds = np.zeros(self.num_types, dtype=np.float32)

        # Ensure type mapping is consistent with Config
        # We assume the dataframe has a 'type' column with string values (e.g., '1JHC')
        # or integer values. If strings, we map them.
        temp_df = df[["type", "scalar_coupling_constant"]].copy()
        if temp_df["type"].dtype == "object":
            temp_df["type"] = temp_df["type"].map(Config.COUPLING_TYPE_MAP)

        for type_idx in range(self.num_types):
            subset = temp_df[temp_df["type"] == type_idx]["scalar_coupling_constant"]
            if len(subset) > 0:
                means[type_idx] = subset.mean()
                stds[type_idx] = subset.std()
            else:
                # Fallback for types not present (should not happen in full train)
                means[type_idx] = 0.0
                stds[type_idx] = 1.0

        # Save to cache
        stats = {"means": means, "stds": stds}
        np.save(self.stats_path, stats)
        print(f"Target statistics saved to {self.stats_path}")

        self.means = torch.tensor(means, dtype=torch.float32, device=self.device)
        self.stds = torch.tensor(stds, dtype=torch.float32, device=self.device)

    def transform(self, values, types):
        """
        Standardizes values: z = (y - mean) / std

        Args:
            values (torch.Tensor): Target values of shape (N,).
            types (torch.Tensor): Coupling type indices of shape (N,).

        Returns:
            torch.Tensor: Standardized values.
        """
        if self.means is None or self.stds is None:
            raise RuntimeError("Standardizer must be fit before transform.")

        device_means = self.means.to(values.device)
        device_stds = self.stds.to(values.device)

        batch_means = device_means[types]
        batch_stds = device_stds[types]

        # Add small epsilon to avoid division by zero if std is 0 (unlikely)
        return (values - batch_means) / (batch_stds + 1e-8)

    def inverse_transform(self, values, types):
        """
        Restores values: y = z * std + mean

        Args:
            values (torch.Tensor): Standardized values of shape (N,).
            types (torch.Tensor): Coupling type indices of shape (N,).

        Returns:
            torch.Tensor: Original scale values.
        """
        if self.means is None or self.stds is None:
            raise RuntimeError("Standardizer must be fit before inverse_transform.")

        device_means = self.means.to(values.device)
        device_stds = self.stds.to(values.device)

        batch_means = device_means[types]
        batch_stds = device_stds[types]

        return values * batch_stds + batch_means


def calculate_lmae(predictions, targets, coupling_types):
    """
    Calculates the Log Mean Absolute Error (LMAE) per coupling type and the average.

    Metric: Mean of Log(MAE) across types.
    LMAE_type = log( mean( |y_pred - y_true| ) )
    Score = mean( LMAE_type for all types )

    Args:
        predictions (torch.Tensor or np.array): Predicted values (original scale).
        targets (torch.Tensor or np.array): Ground truth values.
        coupling_types (torch.Tensor or np.array): Coupling type indices.

    Returns:
        float: The average LMAE score.
        dict: Dictionary mapping type names to their individual LMAE scores.
    """
    # Convert to numpy for metric calculation if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(coupling_types, torch.Tensor):
        coupling_types = coupling_types.detach().cpu().numpy()

    per_type_lmae = {}
    lmae_sum = 0.0
    count = 0

    # Inverse map for readable keys
    inv_map = Config.INVERSE_COUPLING_TYPE_MAP

    for type_idx in range(Config.NUM_COUPLING_TYPES):
        mask = coupling_types == type_idx
        if np.sum(mask) > 0:
            type_preds = predictions[mask]
            type_targets = targets[mask]

            mae = np.mean(np.abs(type_preds - type_targets))
            # Use natural log (np.log) as is standard unless specified otherwise.
            # Add epsilon inside log is not standard for this specific metric definition
            # usually, but we ensure MAE > 0. If MAE is 0, log is -inf.
            # Ideally MAE is never exactly 0 in regression.
            # We clip MAE to a very small number to avoid -inf.
            mae = max(mae, 1e-9)

            lmae = np.log(mae)

            type_name = inv_map.get(type_idx, str(type_idx))
            per_type_lmae[type_name] = lmae
            lmae_sum += lmae
            count += 1

    if count == 0:
        return 0.0, {}

    avg_lmae = lmae_sum / count
    return avg_lmae, per_type_lmae


def save_checkpoint(model, optimizer, scheduler, epoch, score, path):
    """
    Saves the model checkpoint.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "score": score,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)
    print(f"Checkpoint saved to {path} (Score: {score})")


def load_checkpoint(
    model, optimizer=None, scheduler=None, path=None, device=Config.DEVICE
):
    """
    Loads the model checkpoint.
    """
    if path is None:
        path = Config.MODEL_SAVE_PATH

    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return None

    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("score", None)


class Logger:
    """
    Simple logger to write to file and console.
    """

    def __init__(self, log_file):
        self.log_file = log_file
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Clear previous log
        with open(self.log_file, "w") as f:
            f.write("")

    def log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
