import os
import random
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from library.config import COUPLING_TO_INT, NUM_COUPLING_TYPES, DEVICE


def set_seed(seed: int = 42):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


class TargetScaler:
    """
    Handles normalization and denormalization of scalar coupling constants
    separately for each coupling type.
    """

    def __init__(self, device=DEVICE):
        self.device = device
        self.means = torch.zeros(NUM_COUPLING_TYPES, device=device)
        self.stds = torch.ones(NUM_COUPLING_TYPES, device=device)
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Computes mean and std for each coupling type from the provided DataFrame.
        """
        # Calculate stats per type
        stats = df.groupby("type")["scalar_coupling_constant"].agg(["mean", "std"])

        # Fill tensors
        # We iterate through the config mapping to ensure correct index alignment
        for c_type, type_idx in COUPLING_TO_INT.items():
            if c_type in stats.index:
                self.means[type_idx] = stats.loc[c_type, "mean"]
                self.stds[type_idx] = stats.loc[c_type, "std"]
            else:
                # Fallback if a type is missing in the subset (e.g. debug mode)
                self.means[type_idx] = 0.0
                self.stds[type_idx] = 1.0

        self.fitted = True

    def transform(self, values: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Standardizes values: (value - mean) / std
        """
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before transform.")

        # Gather stats for the specific types in the batch
        batch_means = self.means[types]
        batch_stds = self.stds[types]

        # Ensure shapes match for broadcasting if values is (N, 1)
        if values.dim() > 1 and batch_means.dim() == 1:
            batch_means = batch_means.view(-1, 1)
            batch_stds = batch_stds.view(-1, 1)

        return (values - batch_means) / batch_stds

    def inverse_transform(
        self, scaled_values: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverts standardization: scaled_value * std + mean
        """
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fitted before inverse_transform.")

        batch_means = self.means[types]
        batch_stds = self.stds[types]

        if scaled_values.dim() > 1 and batch_means.dim() == 1:
            batch_means = batch_means.view(-1, 1)
            batch_stds = batch_stds.view(-1, 1)

        return scaled_values * batch_stds + batch_means


def log_mae(y_true: torch.Tensor, y_pred: torch.Tensor, types: torch.Tensor) -> float:
    """
    Calculates the competition metric:
    Log of the Mean Absolute Error, calculated for each scalar coupling type,
    and then averaged across types.

    Metric = Mean( Log( MAE(type_i) ) )
    """
    # Move to CPU and numpy for easier grouping
    y_t = y_true.detach().cpu().numpy().flatten()
    y_p = y_pred.detach().cpu().numpy().flatten()
    t_indices = types.detach().cpu().numpy().flatten()

    # Create a dataframe to leverage groupby
    df = pd.DataFrame({"true": y_t, "pred": y_p, "type_idx": t_indices})

    df["abs_diff"] = np.abs(df["true"] - df["pred"])

    # Calculate MAE per type
    mae_per_type = df.groupby("type_idx")["abs_diff"].mean()

    # Calculate Log of MAE
    # Add a small epsilon just in case MAE is 0 to avoid -inf, though unlikely in regression
    log_mae_per_type = np.log(mae_per_type + 1e-9)

    # Average across types
    return float(log_mae_per_type.mean())


class MetricLogger:
    """
    Simple utility to track and print metrics.
    """

    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, metric_name: str, value: float):
        self.metrics[metric_name].append(value)

    def get_average(self, metric_name: str) -> float:
        vals = self.metrics[metric_name]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def reset(self):
        self.metrics.clear()

    def print_metrics(self, prefix: str = ""):
        msg = []
        for k, v in self.metrics.items():
            # If it's a list of values (e.g. batch losses), print the average
            avg_val = sum(v) / len(v)
            msg.append(f"{k}: {avg_val}")

        print(f"{prefix} " + " | ".join(msg))
