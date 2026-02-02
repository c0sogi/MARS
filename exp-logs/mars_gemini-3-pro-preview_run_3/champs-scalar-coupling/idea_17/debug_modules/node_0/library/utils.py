import os
import json
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the centralized configuration method.
    """
    Config.set_seed(seed)


class TargetStandardizer:
    """
    Handles the per-coupling-type standardization (Z-score normalization) of target values.
    Crucial for handling the varying scales of different coupling constants (e.g., 1JHC vs 2JHH).
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.means = None
        self.stds = None
        self.stats_dict = {}

    def fit(self, df: pd.DataFrame = None, load_cached_data: bool = True):
        """
        Computes mean and standard deviation for each coupling type.

        Implements caching logic:
        1. If load_cached_data is True, attempts to load stats from disk.
        2. If loading fails or is disabled, computes stats from the provided DataFrame.
        3. Saves computed stats to disk for future runs.

        Args:
            df: Pandas DataFrame containing 'type' and 'scalar_coupling_constant'.
                Required if cache is not used or missing.
            load_cached_data: Boolean flag to enable/disable loading from cache.
        """
        cache_path = os.path.join(Config.CACHE_DIR, "target_stats.json")
        loaded = False

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.stats_dict = json.load(f)
                loaded = True
            except Exception:
                # If load fails, proceed to compute
                loaded = False

        # 2. Compute from scratch if not loaded
        if not loaded:
            if df is None:
                raise ValueError(
                    "DataFrame is required to fit TargetStandardizer when cache is missing or disabled."
                )

            # Group by coupling type and calculate mean/std
            stats = df.groupby("type")["scalar_coupling_constant"].agg(["mean", "std"])
            self.stats_dict = stats.to_dict(orient="index")

            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self.stats_dict, f, indent=4)

        # 3. Prepare tensors for efficient GPU lookup
        # Create vectors of size NUM_COUPLING_TYPES
        mean_vec = np.zeros(Config.NUM_COUPLING_TYPES, dtype=np.float32)
        std_vec = np.ones(Config.NUM_COUPLING_TYPES, dtype=np.float32)

        for type_name, stats in self.stats_dict.items():
            if type_name in Config.TYPE_TO_IDX:
                idx = Config.TYPE_TO_IDX[type_name]
                mean_vec[idx] = stats["mean"]
                std_vec[idx] = stats["std"]

        self.means = torch.tensor(mean_vec, device=self.device)
        self.stds = torch.tensor(std_vec, device=self.device)

        # Update Config placeholders for reference
        Config.COUPLING_TYPE_MEAN = self.stats_dict
        Config.COUPLING_TYPE_STD = self.stats_dict

    def transform(self, values: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Standardizes values: z = (y - mu) / sigma

        Args:
            values: Tensor of target values (N,)
            types: Tensor of coupling type indices (N,)
        """
        if self.means is None:
            raise RuntimeError(
                "TargetStandardizer must be fit before calling transform."
            )

        types = types.long()
        return (values - self.means[types]) / self.stds[types]

    def inverse_transform(
        self, values: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverses standardization: y = z * sigma + mu

        Args:
            values: Tensor of standardized values (N,)
            types: Tensor of coupling type indices (N,)
        """
        if self.means is None:
            raise RuntimeError(
                "TargetStandardizer must be fit before calling inverse_transform."
            )

        types = types.long()
        return values * self.stds[types] + self.means[types]


class MetricLogger:
    """
    Tracks and computes the Log Mean Absolute Error (LMAE).
    Metric Definition: Log of the Mean Absolute Error, calculated for each scalar coupling type,
    and then averaged across types.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets internal accumulators."""
        self.type_ae_sum = {i: 0.0 for i in range(Config.NUM_COUPLING_TYPES)}
        self.type_count = {i: 0 for i in range(Config.NUM_COUPLING_TYPES)}

    def update(self, preds: torch.Tensor, targets: torch.Tensor, types: torch.Tensor):
        """
        Updates the metric with a batch of predictions.

        Args:
            preds: Predicted values (N,)
            targets: Ground truth values (N,)
            types: Coupling type indices (N,)
        """
        # Move to CPU for accumulation to save GPU memory
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        types = types.detach().cpu().long()

        abs_errors = torch.abs(preds - targets)

        # Iterate over unique types present in the batch for efficiency
        unique_types = torch.unique(types)
        for t_idx in unique_types:
            t_val = t_idx.item()
            mask = types == t_idx

            sum_ae = abs_errors[mask].sum().item()
            count = mask.sum().item()

            self.type_ae_sum[t_val] += sum_ae
            self.type_count[t_val] += count

    def compute(self):
        """
        Computes the final LMAE metric.

        Returns:
            dict: Contains 'LMAE' (scalar) and 'per_type' (dict of scores).
        """
        per_type_lmae = {}
        total_lmae = 0.0
        valid_types_count = 0

        # Helper to get type name from index
        idx_to_type = {v: k for k, v in Config.TYPE_TO_IDX.items()}

        for i in range(Config.NUM_COUPLING_TYPES):
            count = self.type_count[i]
            if count > 0:
                mae = self.type_ae_sum[i] / count
                # Metric is log(MAE). Add small epsilon to prevent log(0)
                lmae = np.log(mae + 1e-9)

                type_name = idx_to_type.get(i, str(i))
                per_type_lmae[type_name] = lmae

                total_lmae += lmae
                valid_types_count += 1

        if valid_types_count > 0:
            avg_lmae = total_lmae / valid_types_count
        else:
            avg_lmae = 0.0

        return {"LMAE": avg_lmae, "per_type": per_type_lmae}
