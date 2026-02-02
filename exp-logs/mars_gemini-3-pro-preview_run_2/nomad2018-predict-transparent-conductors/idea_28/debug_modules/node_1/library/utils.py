import numpy as np
import os
from library.config import Config


class TargetScaler:
    """
    A utility class for standardizing and inverse-transforming target variables.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, targets):
        """
        Computes mean and std of the targets.

        Args:
            targets (np.ndarray): Array of shape (N, num_targets).
        """
        self.mean = np.mean(targets, axis=0)
        self.std = np.std(targets, axis=0)
        # Prevent division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, targets):
        """
        Standardizes the targets using the fitted mean and std.

        Args:
            targets (np.ndarray): Array of shape (N, num_targets).

        Returns:
            np.ndarray: Scaled targets.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted.")
        return (targets - self.mean) / self.std

    def inverse_transform(self, scaled_targets):
        """
        Inverse transforms the scaled targets back to the original scale.

        Args:
            scaled_targets (np.ndarray): Array of shape (N, num_targets).

        Returns:
            np.ndarray: Original scale targets.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted.")
        return (scaled_targets * self.std) + self.mean

    def save(self, path):
        """
        Saves the scaler parameters to a file using numpy format.

        Args:
            path (str): File path to save the scaler.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)
        print(f"TargetScaler saved to {path}")

    def load(self, path):
        """
        Loads the scaler parameters from a file.

        Args:
            path (str): File path to load the scaler from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        print(f"TargetScaler loaded from {path}")


def get_scaler(train_targets, load_cached_data=True):
    """
    Gets a fitted TargetScaler, either by loading from cache or fitting on provided targets.

    Args:
        train_targets (np.ndarray): Array of shape (N, num_targets) to fit on.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        TargetScaler: A fitted scaler instance.
    """
    scaler = TargetScaler()
    cache_path = Config.TARGET_SCALER_CACHE

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            scaler.load(cache_path)
            return scaler
        except Exception as e:
            print(f"Failed to load scaler from cache: {e}. Fitting from scratch.")

    # Fit if not loaded or loading failed
    print("Fitting TargetScaler...")
    scaler.fit(train_targets)
    scaler.save(cache_path)
    return scaler


def compute_metric(y_true, y_pred):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (np.ndarray): Ground truth values, shape (N, num_targets).
        y_pred (np.ndarray): Predicted values, shape (N, num_targets).

    Returns:
        float: The mean RMSLE across all target columns.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure non-negative values for log as targets (energy) should be non-negative
    # and predictions might slightly dip below zero due to model noise.
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    # Calculate squared log error for each element
    # log1p(x) = log(x + 1)
    squared_log_error = (np.log1p(y_true) - np.log1p(y_pred)) ** 2

    # Compute mean squared log error for each column
    msle_per_column = np.mean(squared_log_error, axis=0)

    # Compute RMSLE for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Return the average RMSLE across columns
    return np.mean(rmsle_per_column)
