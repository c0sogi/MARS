import os
import json
import random
import numpy as np
import torch
import pandas as pd
from library import config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def calculate_global_stats(images):
    """
    Calculates global min and max values for each channel across the provided images.
    Used for global normalization.

    Args:
        images (np.ndarray): Array of shape (N, 3, H, W)

    Returns:
        dict: Dictionary containing min and max for each channel.
    """
    stats = {}
    # Iterate over the 3 channels
    for c in range(images.shape[1]):
        stats[f"min_ch{c}"] = float(np.min(images[:, c, :, :]))
        stats[f"max_ch{c}"] = float(np.max(images[:, c, :, :]))
    return stats


def load_and_process_data(load_cached_data=True):
    """
    Loads training and test data from JSON files, processes them into 3-channel images,
    handles missing metadata, and computes global statistics.

    Implements caching to ./working/processed_data.npz to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary containing processed arrays for train/test images,
              targets, incidence angles, IDs, and global statistics.
    """
    cache_path = config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "train_images": data["train_images"],
                "train_targets": data["train_targets"],
                "train_inc_angles": data["train_inc_angles"],
                "train_ids": data["train_ids"],
                "test_images": data["test_images"],
                "test_inc_angles": data["test_inc_angles"],
                "test_ids": data["test_ids"],
                "stats": data["stats"].item(),
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch...")

    # 2. Process from scratch
    print("Processing raw JSON data...")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load raw JSONs
    with open(config.TRAIN_JSON, "r") as f:
        train_raw = json.load(f)
    with open(config.TEST_JSON, "r") as f:
        test_raw = json.load(f)

    def _process_records(records, is_train=True):
        ids = []
        inc_angles = []
        bands_1 = []
        bands_2 = []
        targets = []

        for item in records:
            ids.append(item["id"])

            # Handle incidence angle
            ia = item["inc_angle"]
            if ia == "na":
                inc_angles.append(np.nan)
            else:
                inc_angles.append(float(ia))

            # Process Bands (flattened list -> 75x75)
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
            bands_1.append(b1)
            bands_2.append(b2)

            if is_train:
                targets.append(item["is_iceberg"])

        # Stack into arrays
        ids = np.array(ids)
        inc_angles = np.array(inc_angles, dtype=np.float32)
        b1_stack = np.stack(bands_1)
        b2_stack = np.stack(bands_2)

        # Construct 3rd Channel: Mean of B1 and B2
        b3_stack = (b1_stack + b2_stack) / 2.0

        # Stack channels: (N, 3, 75, 75)
        images = np.stack([b1_stack, b2_stack, b3_stack], axis=1)

        if is_train:
            targets = np.array(targets, dtype=np.float32)
            return ids, inc_angles, images, targets
        else:
            return ids, inc_angles, images

    # Process Train and Test
    train_ids, train_inc, train_imgs, train_targets = _process_records(
        train_raw, is_train=True
    )
    test_ids, test_inc, test_imgs = _process_records(test_raw, is_train=False)

    # 3. Handle Missing Incidence Angles (Mean Imputation)
    # Calculate mean from valid training data
    valid_mask = ~np.isnan(train_inc)
    inc_mean = np.mean(train_inc[valid_mask])

    # Fill NaNs
    train_inc[np.isnan(train_inc)] = inc_mean
    test_inc[np.isnan(test_inc)] = inc_mean

    # 4. Calculate Global Statistics (from Training set only)
    stats = calculate_global_stats(train_imgs)

    # 5. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    np.savez(
        cache_path,
        train_images=train_imgs,
        train_targets=train_targets,
        train_inc_angles=train_inc,
        train_ids=train_ids,
        test_images=test_imgs,
        test_inc_angles=test_inc,
        test_ids=test_ids,
        stats=stats,
    )

    return {
        "train_images": train_imgs,
        "train_targets": train_targets,
        "train_inc_angles": train_inc,
        "train_ids": train_ids,
        "test_images": test_imgs,
        "test_inc_angles": test_inc,
        "test_ids": test_ids,
        "stats": stats,
    }
