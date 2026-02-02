import os
import json
import random
import numpy as np
import torch
from library.config import TRAIN_JSON, CACHE_DIR, SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
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


def calculate_global_stats(load_cached_data=True):
    """
    Calculates global min and max for Band 1, Band 2, and Band 3 (Mean)
    across the entire training dataset.

    The stats are saved as a numpy array in the order:
    [b1_min, b1_max, b2_min, b2_max, b3_min, b3_max]

    Args:
        load_cached_data (bool): If True, attempts to load stats from cache.

    Returns:
        dict: Dictionary containing min and max for each channel.
              Keys: 'b1_min', 'b1_max', 'b2_min', 'b2_max', 'b3_min', 'b3_max'
    """
    stats_cache_path = os.path.join(CACHE_DIR, "global_stats.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(stats_cache_path):
        try:
            # Load as a standard numpy array (no pickle required for numeric arrays)
            stats_arr = np.load(stats_cache_path)
            if stats_arr.shape == (6,):
                return {
                    "b1_min": float(stats_arr[0]),
                    "b1_max": float(stats_arr[1]),
                    "b2_min": float(stats_arr[2]),
                    "b2_max": float(stats_arr[3]),
                    "b3_min": float(stats_arr[4]),
                    "b3_max": float(stats_arr[5]),
                }
        except Exception:
            # If loading fails (corrupt file, etc.), proceed to recompute
            pass

    # 2. Compute from scratch
    # We must load the raw JSON to get the pixel values
    with open(TRAIN_JSON, "r") as f:
        data = json.load(f)

    # Extract bands
    # data is a list of dicts, each has 'band_1' and 'band_2' as lists of floats
    b1_list = [d["band_1"] for d in data]
    b2_list = [d["band_2"] for d in data]

    # Convert to numpy arrays for efficient computation
    # Shape will be (N_samples, 5625)
    b1_arr = np.array(b1_list, dtype=np.float32)
    b2_arr = np.array(b2_list, dtype=np.float32)

    # Calculate Band 3 (Mean of Band 1 and Band 2)
    b3_arr = (b1_arr + b2_arr) / 2.0

    # Compute global stats
    b1_min = np.min(b1_arr)
    b1_max = np.max(b1_arr)
    b2_min = np.min(b2_arr)
    b2_max = np.max(b2_arr)
    b3_min = np.min(b3_arr)
    b3_max = np.max(b3_arr)

    # Pack into array for caching
    stats_arr = np.array(
        [b1_min, b1_max, b2_min, b2_max, b3_min, b3_max], dtype=np.float32
    )

    # 3. Save to cache
    os.makedirs(os.path.dirname(stats_cache_path), exist_ok=True)
    np.save(stats_cache_path, stats_arr)

    return {
        "b1_min": float(b1_min),
        "b1_max": float(b1_max),
        "b2_min": float(b2_min),
        "b2_max": float(b2_max),
        "b3_min": float(b3_min),
        "b3_max": float(b3_max),
    }
