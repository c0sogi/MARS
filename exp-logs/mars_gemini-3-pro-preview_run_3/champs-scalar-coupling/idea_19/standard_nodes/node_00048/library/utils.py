import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config, TYPE_MAP


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_log_mae(
    y_true: torch.Tensor, y_pred: torch.Tensor, types: torch.Tensor
) -> float:
    """
    Calculates the Log of the Mean Absolute Error, calculated for each scalar coupling type,
    and then averaged across types.

    Args:
        y_true: Tensor of ground truth values.
        y_pred: Tensor of predicted values.
        types: Tensor of coupling type indices (integers).

    Returns:
        float: The Log MAE metric.
    """
    # Ensure inputs are on the same device and detached from graph
    y_true = y_true.detach().flatten()
    y_pred = y_pred.detach().flatten()
    types = types.detach().flatten()

    diff = torch.abs(y_true - y_pred)

    log_maes = []

    # Iterate over all known types to ensure we cover the metric definition
    # However, in a batch, not all types might be present.
    # The metric is usually defined over the whole dataset.
    # For batch-wise approximation, we iterate over unique types present.
    unique_types = torch.unique(types)

    for t in unique_types:
        mask = types == t
        if mask.sum() > 0:
            mae = diff[mask].mean()
            # Log of MAE. Add a small epsilon only if mae is exactly 0 to avoid -inf,
            # though physically unlikely in this regression task.
            log_mae = torch.log(mae + 1e-9)
            log_maes.append(log_mae)

    if len(log_maes) == 0:
        return 0.0

    # Average across types
    return torch.stack(log_maes).mean().item()


class Standardizer:
    """
    Handles per-coupling-type standardization (mean/std scaling) of targets.
    """

    def __init__(self, config: Config):
        self.config = config
        self.stats_path = config.STATS_PATH
        # Shape: (8 types, 2 stats). Column 0: Mean, Column 1: Std
        self.stats = np.zeros((len(TYPE_MAP), 2), dtype=np.float32)
        self.device = config.device
        self.stats_tensor = None

    def fit_or_load(self, df: pd.DataFrame = None, load_cached_data: bool = True):
        """
        Fits the standardizer on the provided DataFrame or loads cached statistics.

        Args:
            df: Training DataFrame containing 'type' (string) and 'scalar_coupling_constant'.
            load_cached_data: If True, attempts to load from disk first.
        """
        if load_cached_data and os.path.exists(self.stats_path):
            try:
                self.load()
                return
            except Exception:
                pass  # Fallback to fit

        if df is None:
            raise ValueError(
                "DataFrame must be provided if cache is not found or load_cached_data is False."
            )

        self.fit(df)
        self.save()

    def fit(self, df: pd.DataFrame):
        """
        Computes mean and std for each coupling type from the DataFrame.
        """
        print("Computing target statistics for standardization...")
        for type_name, type_idx in TYPE_MAP.items():
            subset = df[df["type"] == type_name]["scalar_coupling_constant"]
            if len(subset) > 0:
                mu = subset.mean()
                sigma = subset.std()
                # Avoid division by zero if std is 0 (unlikely)
                if sigma < 1e-9:
                    sigma = 1.0
                self.stats[type_idx, 0] = mu
                self.stats[type_idx, 1] = sigma
            else:
                # Default to identity if type not found (should not happen in full train)
                self.stats[type_idx, 0] = 0.0
                self.stats[type_idx, 1] = 1.0

        self._to_tensor()

    def save(self):
        """Saves statistics to disk."""
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        np.save(self.stats_path, self.stats)
        print(f"Standardizer stats saved to {self.stats_path}")

    def load(self):
        """Loads statistics from disk."""
        self.stats = np.load(self.stats_path)
        self._to_tensor()
        print(f"Standardizer stats loaded from {self.stats_path}")

    def _to_tensor(self):
        """Moves stats to the configured device as a tensor."""
        self.stats_tensor = torch.tensor(
            self.stats, dtype=torch.float32, device=self.device
        )

    def transform(self, values: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Standardizes values: z = (y - mu) / sigma

        Args:
            values: Tensor of shape (N,)
            types: Tensor of shape (N,) containing integer type indices
        """
        if self.stats_tensor is None:
            self._to_tensor()

        # Gather mu and sigma for each element based on type
        # stats_tensor shape: (8, 2)
        # types shape: (N,)

        mu = self.stats_tensor[types, 0]
        sigma = self.stats_tensor[types, 1]

        return (values - mu) / sigma

    def inverse_transform(
        self, values: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse standardizes values: y = z * sigma + mu

        Args:
            values: Tensor of shape (N,)
            types: Tensor of shape (N,) containing integer type indices
        """
        if self.stats_tensor is None:
            self._to_tensor()

        mu = self.stats_tensor[types, 0]
        sigma = self.stats_tensor[types, 1]

        return values * sigma + mu
