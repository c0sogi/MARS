import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


class TargetScaler:
    """
    Standardizes the target variable (scalar_coupling_constant) individually for each
    coupling type. Stores statistics to ensure consistent scaling between training
    and inference.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}
        self.types = []

    def fit(
        self,
        df=None,
        target_col="scalar_coupling_constant",
        type_col="type",
        load_cache=True,
    ):
        """
        Computes mean and std for each coupling type.

        Args:
            df (pd.DataFrame): Training metadata containing target and type columns.
            target_col (str): Name of target column.
            type_col (str): Name of type column.
            load_cache (bool): Whether to attempt loading from cache.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.CACHE_STATS_PATH), exist_ok=True)

        # 1. Try to load from cache
        if load_cache and os.path.exists(Config.CACHE_STATS_PATH):
            try:
                print(f"Loading target scaler stats from {Config.CACHE_STATS_PATH}...")
                data = np.load(Config.CACHE_STATS_PATH, allow_pickle=True)
                # .item() is needed because np.savez saves dicts as object arrays
                self.means = data["means"].item()
                self.stds = data["stds"].item()
                self.types = sorted(list(self.means.keys()))
                return
            except Exception as e:
                print(f"Failed to load cached stats: {e}. Recomputing...")

        # 2. Compute from scratch
        if df is None:
            raise ValueError(
                "DataFrame must be provided if cache is not found or load_cache=False."
            )

        print("Computing target scaler stats from data...")
        groups = df.groupby(type_col)[target_col]
        self.means = groups.mean().to_dict()
        self.stds = groups.std().to_dict()
        self.types = sorted(list(self.means.keys()))

        # 3. Save to cache
        print(f"Saving target scaler stats to {Config.CACHE_STATS_PATH}...")
        np.savez(Config.CACHE_STATS_PATH, means=self.means, stds=self.stds)

    def transform(self, df, target_col="scalar_coupling_constant", type_col="type"):
        """
        Standardizes the target column in the DataFrame.
        Returns a numpy array of scaled values.

        Formula: (x - mean) / std
        """
        # Map means and stds to the dataframe rows
        # Using map is efficient for pandas series
        means = df[type_col].map(self.means).values
        stds = df[type_col].map(self.stds).values

        values = df[target_col].values

        # Standardize: (x - mean) / std
        # Add epsilon to std to avoid division by zero
        scaled_values = (values - means) / (stds + 1e-8)

        return scaled_values.astype(np.float32)

    def inverse_transform(self, predictions, types):
        """
        Reverts the standardization to get original scale predictions.

        Args:
            predictions (np.ndarray or torch.Tensor): Scaled predictions.
            types (list or np.ndarray): Coupling types corresponding to predictions.

        Returns:
            np.ndarray: Predictions in original scale.
        """
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()

        # Ensure predictions is flat if it's (N, 1)
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        # Convert types to pandas Series for mapping
        type_series = pd.Series(types)

        means = type_series.map(self.means).values
        stds = type_series.map(self.stds).values

        # Inverse: x * std + mean
        unscaled_values = predictions * stds + means

        return unscaled_values


class LogMAE:
    """
    Competition Metric: Log of the Mean Absolute Error.
    Calculated for each scalar coupling type, and then averaged across types.
    """

    @staticmethod
    def score(y_true, y_pred, types):
        """
        Computes the metric.

        Args:
            y_true (np.ndarray or torch.Tensor): Ground truth values (original scale).
            y_pred (np.ndarray or torch.Tensor): Predicted values (original scale).
            types (list or np.ndarray): Coupling types.

        Returns:
            float: The final LogMAE score.
            dict: Dictionary containing LogMAE per coupling type.
        """
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()

        # Ensure flat arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Create a DataFrame for easy grouping
        df = pd.DataFrame({"type": types, "true": y_true, "pred": y_pred})

        # Calculate Absolute Error
        df["abs_error"] = np.abs(df["true"] - df["pred"])

        # Calculate Mean Absolute Error per type
        mae_per_type = df.groupby("type")["abs_error"].mean()

        # Calculate Log of MAE
        # Use a small epsilon inside log only if MAE is exactly 0 to avoid -inf
        log_mae_per_type = np.log(mae_per_type + 1e-9)

        # Average across types
        final_score = log_mae_per_type.mean()

        return final_score, log_mae_per_type.to_dict()
