import os
import json
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility using the Config method.
    """
    Config.seed_everything(seed)


class MetricMonitor:
    """
    A class to track and average metrics like Loss and F1 score during training.
    """

    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.metrics = {}

    def reset(self):
        """Resets the metric state."""
        self.metrics = {}

    def update(self, metric_name, val, n=1):
        """
        Updates the running average for a given metric.

        Args:
            metric_name (str): Name of the metric.
            val (float): Value to update.
            n (int): Number of samples corresponding to this value (default 1).
        """
        val = float(val)
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"sum": 0.0, "count": 0}
        self.metrics[metric_name]["sum"] += val * n
        self.metrics[metric_name]["count"] += n

    def get_avg(self, metric_name):
        """Returns the current average of the metric."""
        if metric_name not in self.metrics:
            return 0.0
        return self.metrics[metric_name]["sum"] / self.metrics[metric_name]["count"]

    def __str__(self):
        """Returns a formatted string of all tracked metrics."""
        return " | ".join(
            [
                "{}: {:.{prec}f}".format(
                    metric_name, self.get_avg(metric_name), prec=self.float_precision
                )
                for metric_name in self.metrics
            ]
        )


def get_class_mappings(load_cached_data=True):
    """
    Generates or loads class mappings (category_id <-> model_index).

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        class_to_idx (dict): Maps category_id (int) to model index (int).
        idx_to_class (dict): Maps model index (int) to category_id (int).
    """
    mapping_path = os.path.join(Config.WORKING_DIR, "class_mappings.json")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r") as f:
                mappings = json.load(f)
                # JSON keys are always strings, convert them back to integers
                class_to_idx = {int(k): v for k, v in mappings["class_to_idx"].items()}
                idx_to_class = {int(k): v for k, v in mappings["idx_to_class"].items()}
                return class_to_idx, idx_to_class
        except Exception as e:
            print(f"Failed to load cached mappings: {e}. Recomputing...")

    # Compute mappings from scratch
    df = pd.read_csv(Config.TRAIN_CSV)
    unique_classes = sorted(df["category_id"].unique())

    class_to_idx = {int(c): i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: int(c) for i, c in enumerate(unique_classes)}

    # Save to cache
    with open(mapping_path, "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f)

    return class_to_idx, idx_to_class


def get_class_weights(load_cached_data=True):
    """
    Calculates inverse frequency weights for the classes to handle imbalance.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        torch.Tensor: Weights of shape (num_classes,).
    """
    weights_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(weights_path):
        try:
            weights = np.load(weights_path)
            return torch.from_numpy(weights).float()
        except Exception as e:
            print(f"Failed to load cached weights: {e}. Recomputing...")

    # Load data and ensure mappings exist
    df = pd.read_csv(Config.TRAIN_CSV)
    class_to_idx, _ = get_class_mappings(load_cached_data=load_cached_data)

    # Calculate counts aligned with model indices
    counts_series = df["category_id"].value_counts()
    num_classes = len(class_to_idx)
    counts = np.zeros(num_classes)

    for cat_id, count in counts_series.items():
        if cat_id in class_to_idx:
            idx = class_to_idx[cat_id]
            counts[idx] = count

    # Compute inverse frequency weights: Total / (Num_Classes * Count)
    # Add epsilon to count to avoid division by zero
    total_samples = len(df)
    weights = total_samples / (num_classes * (counts + 1e-6))

    # Save to cache
    np.save(weights_path, weights)

    return torch.from_numpy(weights).float()
