import os
import json
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_global_stats(load_cached_data=True):
    """
    Computes (or loads) the global minimum and maximum values for Band 1 and Band 2
    across the entire training dataset. These stats are used for global min-max normalization.

    Args:
        load_cached_data (bool): If True, attempts to load stats from cache.
                                 If False, forces re-computation.

    Returns:
        dict: A dictionary containing 'band_1_min', 'band_1_max', 'band_2_min', 'band_2_max'.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "global_stats.json")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                stats = json.load(f)
            return stats
        except (json.JSONDecodeError, IOError):
            # If cache is corrupt, proceed to recompute
            pass

    # 2. Compute from scratch
    # Ensure directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load raw training data
    # We use the raw json because the metadata CSV doesn't contain the band data
    if not os.path.exists(Config.TRAIN_JSON):
        raise FileNotFoundError(f"Training data not found at {Config.TRAIN_JSON}")

    with open(Config.TRAIN_JSON, "r") as f:
        data = json.load(f)

    # Initialize min/max trackers
    b1_min = float("inf")
    b1_max = float("-inf")
    b2_min = float("inf")
    b2_max = float("-inf")

    # Iterate through all samples
    for item in data:
        # Band 1
        b1 = np.array(item["band_1"], dtype=np.float32)
        current_b1_min = float(np.min(b1))
        current_b1_max = float(np.max(b1))

        if current_b1_min < b1_min:
            b1_min = current_b1_min
        if current_b1_max > b1_max:
            b1_max = current_b1_max

        # Band 2
        b2 = np.array(item["band_2"], dtype=np.float32)
        current_b2_min = float(np.min(b2))
        current_b2_max = float(np.max(b2))

        if current_b2_min < b2_min:
            b2_min = current_b2_min
        if current_b2_max > b2_max:
            b2_max = current_b2_max

    stats = {
        "band_1_min": b1_min,
        "band_1_max": b1_max,
        "band_2_min": b2_min,
        "band_2_max": b2_max,
    }

    # 3. Save to cache
    with open(cache_file, "w") as f:
        json.dump(stats, f, indent=4)

    return stats
