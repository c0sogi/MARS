import os
import json
import random
import numpy as np
import torch
import pandas as pd
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


def get_device():
    """
    Returns the PyTorch device to be used (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _process_json_data(json_path, is_train=True):
    """
    Helper function to process raw JSON data into numpy arrays.

    Args:
        json_path (str): Path to the JSON file.
        is_train (bool): Whether the file is training data (contains labels).

    Returns:
        dict: Dictionary containing processed images, angles, ids, and optional labels.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 and Band 2 are lists of floats
    band_1 = np.stack([np.array(b).reshape(75, 75) for b in df["band_1"].values])
    band_2 = np.stack([np.array(b).reshape(75, 75) for b in df["band_2"].values])

    # Construct 3rd Channel: Mean of Band 1 and Band 2
    band_3 = (band_1 + band_2) / 2.0

    # Stack into (N, 3, 75, 75)
    # Using axis 1 for channels: (N, C, H, W)
    images = np.stack([band_1, band_2, band_3], axis=1).astype(np.float32)

    # Process Incidence Angle
    # Replace 'na' with NaN and convert to float
    inc_angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(
        np.float32
    )

    ids = df["id"].values

    result = {"images": images, "angles": inc_angles, "ids": ids}

    if is_train and "is_iceberg" in df.columns:
        result["labels"] = df["is_iceberg"].values.astype(np.float32)

    return result


def load_data(config, load_cached_data=True):
    """
    Loads dataset, using a cached .npz file if available and requested.
    Otherwise, processes raw JSON files, creates the cache, and returns data.

    Args:
        config (Config): Configuration object containing paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'train_images', 'train_angles', 'train_labels', 'train_ids',
              'test_images', 'test_angles', 'test_ids'.
    """
    cache_path = config.CACHE_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading data from cache: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)

            # Validate cache schema
            required_keys = [
                "train_images",
                "train_angles",
                "train_labels",
                "train_ids",
                "test_images",
                "test_angles",
                "test_ids",
            ]
            if not all(key in cached.files for key in required_keys):
                raise KeyError(f"Cache missing required keys. Found: {cached.files}")

            return {key: cached[key] for key in cached.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process data from scratch
    print("Processing raw data from JSON files...")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Process Train
    train_data = _process_json_data(config.TRAIN_JSON, is_train=True)

    # Process Test
    test_data = _process_json_data(config.TEST_JSON, is_train=False)

    # Prepare dictionary for saving/returning
    data_dict = {
        "train_images": train_data["images"],
        "train_angles": train_data["angles"],
        "train_labels": train_data["labels"],
        "train_ids": train_data["ids"],
        "test_images": test_data["images"],
        "test_angles": test_data["angles"],
        "test_ids": test_data["ids"],
    }

    # 3. Save to cache
    print(f"Saving processed data to cache: {cache_path}")
    np.savez_compressed(cache_path, **data_dict)

    return data_dict
