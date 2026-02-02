import os
import json
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_global_stats(load_cached_data=True):
    """
    Calculates or loads the global minimum and maximum values for Band 1 and Band 2
    from the entire training dataset. These stats are used for global min-max scaling.

    Args:
        load_cached_data (bool): If True, attempts to load stats from a local cache file.

    Returns:
        dict: A dictionary containing 'b1_min', 'b1_max', 'b2_min', 'b2_max'.
    """
    cache_path = os.path.join(Config.WORK_DIR, "global_stats.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading global stats from cache: {cache_path}")
            # We save as a simple array: [b1_min, b1_max, b2_min, b2_max]
            stats_array = np.load(cache_path)
            stats = {
                "b1_min": float(stats_array[0]),
                "b1_max": float(stats_array[1]),
                "b2_min": float(stats_array[2]),
                "b2_max": float(stats_array[3]),
            }
            return stats
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing global stats from raw training data...")
    train_json_path = os.path.join(Config.INPUT_DIR, "train.json")

    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Training data not found at {train_json_path}")

    with open(train_json_path, "r") as f:
        data = json.load(f)

    # Initialize with infinity
    b1_min = float("inf")
    b1_max = float("-inf")
    b2_min = float("inf")
    b2_max = float("-inf")

    # Iterate to find min/max
    # Note: Loading all into a giant numpy array might be memory intensive,
    # but iterating list by list is safe.
    for item in data:
        b1 = np.array(item["band_1"])
        b2 = np.array(item["band_2"])

        b1_min = min(b1_min, b1.min())
        b1_max = max(b1_max, b1.max())

        b2_min = min(b2_min, b2.min())
        b2_max = max(b2_max, b2.max())

    stats = {"b1_min": b1_min, "b1_max": b1_max, "b2_min": b2_min, "b2_max": b2_max}

    print(f"Computed Stats: {stats}")

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    stats_array = np.array([b1_min, b1_max, b2_min, b2_max])
    np.save(cache_path, stats_array)
    print(f"Global stats saved to {cache_path}")

    return stats
