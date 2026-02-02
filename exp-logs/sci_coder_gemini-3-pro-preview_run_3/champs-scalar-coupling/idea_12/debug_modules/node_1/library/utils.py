import os
import random
import json
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    Handles per-coupling-type standardization (mean/std scaling) and inverse transformation.
    This is critical because different coupling types have vastly different ranges.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}
        # Arrays for vectorized operations, indexed by coupling type index
        # Initialize with 0 mean and 1 std (identity transform)
        self.mean_arr = np.zeros(len(Config.COUPLING_TYPES), dtype=np.float32)
        self.std_arr = np.ones(len(Config.COUPLING_TYPES), dtype=np.float32)
        self.fitted = False

    def fit(self, df):
        """
        Computes mean and std for each coupling type from the provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing 'type' and 'scalar_coupling_constant'.
        """
        print("Fitting TargetScaler...")
        for i, coupling_type in enumerate(Config.COUPLING_TYPES):
            subset = df[df["type"] == coupling_type]
            if "scalar_coupling_constant" in subset.columns and len(subset) > 0:
                vals = subset["scalar_coupling_constant"].values
                mu = np.mean(vals)
                sigma = np.std(vals)

                self.means[coupling_type] = float(mu)
                self.stds[coupling_type] = float(sigma)

                self.mean_arr[i] = mu
                self.std_arr[i] = sigma
            else:
                # Fallback if type not present (should not happen in full train)
                self.means[coupling_type] = 0.0
                self.stds[coupling_type] = 1.0
                self.mean_arr[i] = 0.0
                self.std_arr[i] = 1.0

        self.fitted = True
        print("TargetScaler fitted.")

    def transform_df(self, df):
        """
        Standardizes the 'scalar_coupling_constant' column in the DataFrame.
        Returns a new DataFrame with scaled values.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with standardized targets.
        """
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before transform.")

        df_out = df.copy()
        if "scalar_coupling_constant" not in df_out.columns:
            return df_out

        # Map string types to integer indices
        type_indices = df_out["type"].map(Config.COUPLING_MAP).values.astype(int)

        # Extract values
        vals = df_out["scalar_coupling_constant"].values

        # Broadcast means and stds
        means = self.mean_arr[type_indices]
        stds = self.std_arr[type_indices]

        # Scale: (x - mu) / sigma
        df_out["scalar_coupling_constant"] = (vals - means) / stds

        return df_out

    def inverse_transform(self, preds, types):
        """
        Inverse transforms standardized predictions back to the original scale.

        Args:
            preds: Numpy array or Torch tensor of shape (N,) or (N, 1) containing scaled values.
            types: Numpy array or Torch tensor of shape (N,) containing integer indices of coupling types.

        Returns:
            Numpy array of shape (N,) containing values in original physical units.
        """
        # Convert tensors to numpy if necessary
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(types):
            types = types.detach().cpu().numpy()

        preds = preds.flatten()
        types = types.flatten().astype(int)

        # Lookup means and stds for each sample
        means = self.mean_arr[types]
        stds = self.std_arr[types]

        # Inverse: y = z * sigma + mu
        return preds * stds + means

    def save(self, path):
        """Saves the scaler statistics to a JSON file."""
        data = {"means": self.means, "stds": self.stds}
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, path):
        """Loads the scaler statistics from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        with open(path, "r") as f:
            data = json.load(f)
        self.means = data["means"]
        self.stds = data["stds"]

        # Reconstruct arrays
        for i, coupling_type in enumerate(Config.COUPLING_TYPES):
            if coupling_type in self.means:
                self.mean_arr[i] = self.means[coupling_type]
                self.std_arr[i] = self.stds[coupling_type]

        self.fitted = True


class MetricLogger:
    """
    Tracks and calculates the Log Mean Absolute Error (LMAE) metric.
    The metric is defined as the mean of the logarithm of the MAE for each coupling type.
    """

    def __init__(self):
        self.predictions = []
        self.targets = []
        self.types = []

    def reset(self):
        """Clears stored predictions and targets."""
        self.predictions = []
        self.targets = []
        self.types = []

    def update(self, preds, targets, types):
        """
        Updates the logger with a batch of predictions and targets.

        Args:
            preds: Scaled predictions (Tensor or array)
            targets: Scaled targets (Tensor or array)
            types: Coupling type indices (Tensor or array)
        """
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        if torch.is_tensor(types):
            types = types.detach().cpu().numpy()

        self.predictions.append(preds.flatten())
        self.targets.append(targets.flatten())
        self.types.append(types.flatten())

    def compute_metric(self, scaler=None):
        """
        Computes the Log Mean Absolute Error across all accumulated batches.

        Args:
            scaler: Optional TargetScaler instance to inverse transform values before calculation.
                    If None, assumes values are already in original scale.

        Returns:
            float: The LMAE score.
        """
        if not self.predictions:
            return 0.0

        all_preds = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)
        all_types = np.concatenate(self.types).astype(int)

        # Inverse transform if scaler is provided (assuming inputs were scaled)
        if scaler:
            all_preds = scaler.inverse_transform(all_preds, all_types)
            all_targets = scaler.inverse_transform(all_targets, all_types)

        # Calculate metric per type
        maes = []
        for i, type_name in enumerate(Config.COUPLING_TYPES):
            mask = all_types == i
            if np.sum(mask) > 0:
                # MAE for this type
                diff = np.abs(all_preds[mask] - all_targets[mask])
                mae = np.mean(diff)

                # Log of MAE
                # Add small epsilon to avoid log(0) though unlikely physically
                maes.append(np.log(mae + 1e-9))

        if not maes:
            return 0.0

        # Average across types
        return np.mean(maes)

    def get_per_type_metrics(self, scaler=None):
        """
        Returns a dictionary of MAE and LogMAE per coupling type.
        Useful for detailed validation analysis.

        Returns:
            dict: Dictionary mapping coupling type to metrics.
        """
        if not self.predictions:
            return {}

        all_preds = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)
        all_types = np.concatenate(self.types).astype(int)

        if scaler:
            all_preds = scaler.inverse_transform(all_preds, all_types)
            all_targets = scaler.inverse_transform(all_targets, all_types)

        results = {}
        for i, type_name in enumerate(Config.COUPLING_TYPES):
            mask = all_types == i
            if np.sum(mask) > 0:
                diff = np.abs(all_preds[mask] - all_targets[mask])
                mae = np.mean(diff)
                results[type_name] = {
                    "mae": mae,
                    "log_mae": np.log(mae + 1e-9),
                    "count": int(np.sum(mask)),
                }
        return results
