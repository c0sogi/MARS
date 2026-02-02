import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SEED, METRIC_CLIP_SIGMA, METRIC_MAX_DELTA


def seed_everything(seed=SEED):
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


def metric_laplace_log_likelihood(y_true, y_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (numpy array or torch tensor).
        y_pred: Predicted FVC values (numpy array or torch tensor).
        sigma_pred: Predicted Confidence/Sigma values (numpy array or torch tensor).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Convert tensors to numpy arrays for calculation
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Clip sigma to prevent singularity and reflect measurement uncertainty
    sigma_clipped = np.maximum(sigma_pred, METRIC_CLIP_SIGMA)

    # Calculate absolute error and clip it
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, METRIC_MAX_DELTA)

    # Calculate the metric
    # Note: np.log is natural logarithm (ln)
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)


class DataScaler:
    """
    Handles Z-score standardization for FVC, Weeks, Age, and Baseline FVC.
    Stores mean and std from the fitting dataset (training set).
    """

    def __init__(self):
        self.means = {}
        self.stds = {}
        self.fitted = False

    def fit(self, df):
        """
        Computes and stores mean and standard deviation for relevant columns.

        Args:
            df (pd.DataFrame): Training dataframe containing 'FVC', 'Weeks', 'Age'.
        """
        # Target variable FVC
        self.means["FVC"] = df["FVC"].mean()
        self.stds["FVC"] = df["FVC"].std()

        # Features
        self.means["Weeks"] = df["Weeks"].mean()
        self.stds["Weeks"] = df["Weeks"].std()

        self.means["Age"] = df["Age"].mean()
        self.stds["Age"] = df["Age"].std()

        self.fitted = True

    def transform(self, df):
        """
        Applies Z-score scaling to the provided dataframe.

        Args:
            df (pd.DataFrame): Dataframe to transform.

        Returns:
            pd.DataFrame: A copy of the dataframe with scaled columns.
        """
        if not self.fitted:
            raise RuntimeError("DataScaler must be fitted before transform.")

        df_scaled = df.copy()

        # Scale FVC if present (Target)
        if "FVC" in df.columns:
            df_scaled["FVC"] = (df["FVC"] - self.means["FVC"]) / self.stds["FVC"]

        # Scale Weeks (Feature)
        if "Weeks" in df.columns:
            df_scaled["Weeks"] = (df["Weeks"] - self.means["Weeks"]) / self.stds[
                "Weeks"
            ]

        # Scale Age (Feature)
        if "Age" in df.columns:
            df_scaled["Age"] = (df["Age"] - self.means["Age"]) / self.stds["Age"]

        # Scale Baseline FVC (Feature) - Uses FVC statistics
        if "Baseline" in df.columns:
            df_scaled["Baseline"] = (df["Baseline"] - self.means["FVC"]) / self.stds[
                "FVC"
            ]

        return df_scaled

    def transform_value(self, col_name, value):
        """
        Transforms a single value or array based on the stats of col_name.
        Useful for inference when full dataframe is not constructed.

        Args:
            col_name (str): 'FVC', 'Weeks', 'Age', or 'Baseline'.
            value (float or np.array): Value to transform.

        Returns:
            float or np.array: Scaled value.
        """
        if not self.fitted:
            raise RuntimeError("DataScaler must be fitted before transform.")

        # Baseline uses FVC stats
        stats_key = "FVC" if col_name == "Baseline" else col_name

        if stats_key not in self.means:
            raise ValueError(f"No statistics found for column: {col_name}")

        return (value - self.means[stats_key]) / self.stds[stats_key]

    def inverse_transform_target(self, y_scaled):
        """
        Converts scaled FVC predictions back to the original ml scale.

        Args:
            y_scaled (float, np.array, or torch.Tensor): Scaled FVC predictions.

        Returns:
            Unscaled FVC values.
        """
        if not self.fitted:
            raise RuntimeError("DataScaler must be fitted before inverse_transform.")

        return y_scaled * self.stds["FVC"] + self.means["FVC"]

    def inverse_transform_sigma(self, sigma_scaled):
        """
        Converts scaled uncertainty (sigma) back to the original ml scale.
        Note: Uncertainty is a magnitude, so we only multiply by std, we do not add mean.

        Args:
            sigma_scaled (float, np.array, or torch.Tensor): Scaled sigma predictions.

        Returns:
            Unscaled sigma values.
        """
        if not self.fitted:
            raise RuntimeError("DataScaler must be fitted before inverse_transform.")

        return sigma_scaled * self.stds["FVC"]
