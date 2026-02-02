import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from library.config import Config, seed_everything


class MetricMonitor:
    """
    A utility class to track metrics (loss, accuracy, etc.) during training.
    Maintains a running average of the values.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def get_file_sizes(
    df, root_dir=Config.INPUT_DIR, cache_name="file_sizes", load_cached_data=True
):
    """
    Extracts file sizes (in bytes) for images listed in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing a 'file_path' column.
        root_dir (str): Root directory where images are stored.
        cache_name (str): Filename for the cache file (without extension).
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: Array of file sizes.
    """
    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached file sizes from {cache_path}")
        return np.load(cache_path)

    # 2. Compute from scratch
    # print(f"Computing file sizes for {len(df)} images...")
    sizes = []

    # Ensure we use the correct column. Metadata CSVs have 'file_path'.
    if "file_path" not in df.columns:
        # Fallback if 'file_path' is missing but 'id' exists
        # This assumes train images are in 'train/' and test in 'test/'
        # But the metadata should have 'file_path'.
        raise ValueError("DataFrame must contain 'file_path' column.")

    paths = df["file_path"].values

    for rel_path in paths:
        full_path = os.path.join(root_dir, rel_path)
        if os.path.exists(full_path):
            sizes.append(os.path.getsize(full_path))
        else:
            # If file is missing, append 0 (should not happen with valid metadata)
            sizes.append(0)

    sizes = np.array(sizes, dtype=np.float32)

    # 3. Save to cache
    np.save(cache_path, sizes)
    # print(f"Saved file sizes to {cache_path}")

    return sizes


def normalize_file_sizes(train_sizes, val_sizes, test_sizes):
    """
    Normalizes file sizes for use in the model.

    1. Log-transforms all sizes.
    2. Calculates Mean and Std from TRAIN set for Z-score normalization (FiLM inputs).
    3. Calculates Min and Max from TRAIN set for 0-1 scaling (Auxiliary targets).

    Args:
        train_sizes (np.ndarray): File sizes for training set.
        val_sizes (np.ndarray): File sizes for validation set.
        test_sizes (np.ndarray): File sizes for test set.

    Returns:
        dict: Dictionary containing normalized features and targets for all sets.
              Keys: 'train_film', 'train_aux', 'val_film', 'val_aux', 'test_film', 'test_aux'
    """
    # 1. Log Transform (log1p for numerical stability)
    train_log = np.log1p(train_sizes)
    val_log = np.log1p(val_sizes)
    test_log = np.log1p(test_sizes)

    # 2. Calculate Statistics from Training Set
    mean = train_log.mean()
    std = train_log.std()
    min_val = train_log.min()
    max_val = train_log.max()

    # Safety checks for division
    if std == 0:
        std = 1.0
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1.0

    # 3. FiLM Features: Z-score Normalization
    train_film = (train_log - mean) / std
    val_film = (val_log - mean) / std
    test_film = (test_log - mean) / std

    # 4. Aux Targets: Min-Max Scaling (0-1)
    train_aux = (train_log - min_val) / range_val
    val_aux = (val_log - min_val) / range_val
    test_aux = (test_log - min_val) / range_val

    # Clip aux targets to [0, 1] to handle outliers in val/test
    train_aux = np.clip(train_aux, 0.0, 1.0)
    val_aux = np.clip(val_aux, 0.0, 1.0)
    test_aux = np.clip(test_aux, 0.0, 1.0)

    return {
        "train_film": train_film,
        "train_aux": train_aux,
        "val_film": val_film,
        "val_aux": val_aux,
        "test_film": test_film,
        "test_aux": test_aux,
        "stats": {"mean": mean, "std": std, "min": min_val, "max": max_val},
    }
