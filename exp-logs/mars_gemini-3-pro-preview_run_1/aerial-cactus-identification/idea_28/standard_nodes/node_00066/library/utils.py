import os
import random
import logging
import sys
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


def seed_everything(seed=42):
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


def get_logger(name):
    """
    Creates and configures a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handler already exists to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class MetricMonitor:
    """
    A utility class to track metrics (like Loss, Accuracy) during training/validation loops.
    """

    def __init__(self, float_precision=6):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        """
        Update the metric with a new value.
        Args:
            metric_name (str): Name of the metric.
            val (float): The value to add.
        """
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get_avg(self, metric_name):
        return self.metrics[metric_name]["avg"]

    def __str__(self):
        """
        Returns a formatted string of current average metrics.
        """
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, metric["avg"], prec=self.float_precision
                )
                for metric_name, metric in self.metrics.items()
            ]
        )


def extract_file_sizes(
    df, input_dir, cache_name, load_cached_data=True, normalization_stats=None
):
    """
    Extracts file sizes for images in the dataframe, applies log transformation,
    and normalizes them to [0, 1].

    Implements caching to avoid repeated I/O operations.

    Args:
        df (pd.DataFrame): Dataframe containing a 'file_path' column (relative path).
        input_dir (str): Base directory for the images.
        cache_name (str): Filename for the cache file (e.g., 'train_fsizes.npy').
        load_cached_data (bool): Whether to attempt loading from cache.
        normalization_stats (tuple, optional): (min_val, max_val) from training set.
                                               If None, computes from current data.

    Returns:
        np.array: Normalized log-transformed file sizes.
        tuple: (min_val, max_val) used for normalization.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, cache_name)

    log_sizes = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            log_sizes = np.load(cache_path)
            # print(f"Loaded file sizes from cache: {cache_path}")
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}")
            log_sizes = None

    # 2. Compute if not cached
    if log_sizes is None:
        sizes = []
        # Assuming df has 'file_path' as per metadata generation
        # If not, fall back to constructing from 'id'
        if "file_path" in df.columns:
            paths = df["file_path"].values
        else:
            # Fallback based on typical structure if file_path missing
            # But metadata ensures file_path exists.
            # We assume relative paths are correct.
            paths = df["id"].values

        for rel_path in paths:
            full_path = os.path.join(input_dir, rel_path)
            if os.path.exists(full_path):
                sizes.append(os.path.getsize(full_path))
            else:
                # Handle missing files gracefully (though shouldn't happen with clean metadata)
                sizes.append(0)

        sizes = np.array(sizes, dtype=np.float32)

        # Apply Log Transformation: log(size + 1) to handle potential zeros and compress range
        log_sizes = np.log1p(sizes)

        # Save to cache
        np.save(cache_path, log_sizes)
        # print(f"Saved file sizes to cache: {cache_path}")

    # 3. Normalize
    if normalization_stats is None:
        min_val = log_sizes.min()
        max_val = log_sizes.max()
    else:
        min_val, max_val = normalization_stats

    # Avoid division by zero
    denom = max_val - min_val
    if denom == 0:
        denom = 1.0

    normalized_sizes = (log_sizes - min_val) / denom

    # Clip to [0, 1] to ensure stability if test data is outside train range
    normalized_sizes = np.clip(normalized_sizes, 0.0, 1.0)

    return normalized_sizes, (min_val, max_val)
