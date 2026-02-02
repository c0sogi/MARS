import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_log_mae(preds, targets, types):
    """
    Computes the Log of the Mean Absolute Error (LMAE), averaged across coupling types.

    Metric = Mean( Log( MAE(type_i) ) )

    Args:
        preds (np.ndarray or torch.Tensor): Predicted scalar coupling constants.
        targets (np.ndarray or torch.Tensor): Actual scalar coupling constants.
        types (np.ndarray, list, or pd.Series): The coupling type for each sample (e.g., '1JHC').

    Returns:
        float: The calculated metric.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    types = np.array(types).flatten()

    # Create a DataFrame for easy grouping
    data = pd.DataFrame({"pred": preds, "target": targets, "type": types})

    # Calculate Absolute Error
    data["abs_error"] = np.abs(data["pred"] - data["target"])

    # Calculate MAE per type
    mae_per_type = data.groupby("type")["abs_error"].mean()

    # Calculate Log of MAE (using natural log, as is standard unless log10 specified)
    # Note: If MAE is 0, log is -inf. We assume MAE > 0 for practical ML tasks.
    log_mae_per_type = np.log(mae_per_type)

    # Average across types
    metric = log_mae_per_type.mean()

    return float(metric)


class Standardizer:
    """
    Standardizes the target variable (scalar_coupling_constant) by removing the mean
    and scaling to unit variance, calculated independently for each coupling type.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Computes the mean and standard deviation for each coupling type.

        Args:
            df (pd.DataFrame): Training metadata containing 'type' and 'scalar_coupling_constant'.
        """
        # Check required columns
        if "type" not in df.columns or "scalar_coupling_constant" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'type' and 'scalar_coupling_constant' columns."
            )

        # Compute stats
        stats = df.groupby("type")["scalar_coupling_constant"].agg(["mean", "std"])

        # Store as dictionaries for fast mapping
        self.means = stats["mean"].to_dict()
        self.stds = stats["std"].to_dict()
        self.fitted = True

        # Handle potential zero std (unlikely in this dataset but good for robustness)
        for t, s in self.stds.items():
            if s == 0:
                self.stds[t] = 1.0
                # In a real logging scenario, we might warn here

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the standardization to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing 'type' and 'scalar_coupling_constant'.

        Returns:
            pd.DataFrame: A new DataFrame with the transformed target variable.
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before calling transform.")

        df_transformed = df.copy()

        # Map means and stds to the rows based on type
        means_col = df_transformed["type"].map(self.means)
        stds_col = df_transformed["type"].map(self.stds)

        # Check for types not seen during fit
        if means_col.isnull().any():
            missing_types = df_transformed.loc[means_col.isnull(), "type"].unique()
            raise ValueError(f"Found types in data not present in fit: {missing_types}")

        # Standardize: (x - mu) / sigma
        df_transformed["scalar_coupling_constant"] = (
            df_transformed["scalar_coupling_constant"] - means_col
        ) / stds_col

        return df_transformed

    def inverse_transform(self, values, types):
        """
        Reverts the standardization (denormalizes) for predictions.

        Args:
            values (np.ndarray or torch.Tensor): Normalized predicted values.
            types (np.ndarray, list, or pd.Series): Corresponding coupling types.

        Returns:
            np.ndarray: Denormalized values in the original scale.
        """
        if not self.fitted:
            raise RuntimeError(
                "Standardizer must be fitted before calling inverse_transform."
            )

        # Handle Tensor inputs
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        # Ensure 1D arrays
        values = np.array(values).flatten()
        types = np.array(types).flatten()

        if len(values) != len(types):
            raise ValueError(
                f"Length mismatch: values ({len(values)}) vs types ({len(types)})"
            )

        # Use pandas Series for efficient mapping
        type_series = pd.Series(types)

        means_vec = type_series.map(self.means).values
        stds_vec = type_series.map(self.stds).values

        # Check for missing types in map (results in NaN)
        if np.isnan(means_vec).any():
            raise ValueError("Found types in input not present in Standardizer fit.")

        # Inverse: x * sigma + mu
        original_values = values * stds_vec + means_vec

        return original_values
