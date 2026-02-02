import os
import json
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets seeds for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_global_stats(load_cached_data: bool = True, debug: bool = Config.DEBUG):
    """
    Calculates the global minimum and maximum values for Band 1 and Band 2
    across the training dataset. Implements caching to avoid re-computation.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, uses a subset of data (defined by Config.MAX_SAMPLES).

    Returns:
        dict: A dictionary containing 'b1_min', 'b1_max', 'b2_min', 'b2_max'.
    """
    # Determine cache filename based on debug status to prevent mixing full/debug stats
    cache_filename = "global_stats_debug.npz" if debug else "global_stats.npz"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            stats = {
                "b1_min": float(cached["b1_min"]),
                "b1_max": float(cached["b1_max"]),
                "b2_min": float(cached["b2_min"]),
                "b2_max": float(cached["b2_max"]),
            }
            print(f"Loaded global stats from {cache_path}")
            return stats
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing global stats from training data...")

    # Load raw training data
    if not os.path.exists(Config.TRAIN_JSON):
        raise FileNotFoundError(f"Training data not found at {Config.TRAIN_JSON}")

    with open(Config.TRAIN_JSON, "r") as f:
        data = json.load(f)

    # Handle Debugging/Subsetting
    if debug and Config.MAX_SAMPLES is not None:
        data = data[: Config.MAX_SAMPLES]
        print(f"Debug mode: Computed stats on {len(data)} samples.")

    # Extract bands
    # band_1 and band_2 are lists of floats in the JSON structure
    band_1_list = [item["band_1"] for item in data]
    band_2_list = [item["band_2"] for item in data]

    # Convert to numpy for efficient min/max calculation
    # Flattening happens implicitly if we just feed the list of lists to np.array
    b1_arr = np.array(band_1_list, dtype=np.float32)
    b2_arr = np.array(band_2_list, dtype=np.float32)

    stats = {
        "b1_min": float(np.min(b1_arr)),
        "b1_max": float(np.max(b1_arr)),
        "b2_min": float(np.min(b2_arr)),
        "b2_max": float(np.max(b2_arr)),
    }

    # 3. Save to cache
    np.savez(
        cache_path,
        b1_min=stats["b1_min"],
        b1_max=stats["b1_max"],
        b2_min=stats["b2_min"],
        b2_max=stats["b2_max"],
    )
    print(f"Saved global stats to {cache_path}")
    print(f"Stats: {stats}")

    return stats
