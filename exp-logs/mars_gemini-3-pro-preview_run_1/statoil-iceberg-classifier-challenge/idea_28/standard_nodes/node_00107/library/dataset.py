import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import compute_global_stats

# Constants derived from Data Analysis
INC_ANGLE_MEAN = 39.2829
INC_ANGLE_STD = 3.8362


def load_and_process_data(metadata_path, json_path, cache_name, load_cached_data=True):
    """
    Loads data based on metadata, processing raw JSON if cache is not available.
    Strictly follows the caching logic requirement.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["images"],
                data["angles"],
                data["labels"] if "labels" in data else None,
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute/process from scratch
    # print(f"Processing data from {json_path} using metadata {metadata_path}...")

    # Load metadata to know which samples to extract
    df_meta = pd.read_csv(metadata_path)

    # Load raw JSON
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Map raw data by ID for O(1) access or use sample_index if aligned
    # The metadata contains 'sample_index' which refers to the index in the raw json list.
    # We can use that directly for speed.
    indices = df_meta["sample_index"].values
    ids = df_meta["id"].values

    # Pre-allocate arrays
    n_samples = len(df_meta)
    images = np.zeros((n_samples, 75, 75, 2), dtype=np.float32)
    angles = np.zeros((n_samples,), dtype=np.float32)
    has_labels = "is_iceberg" in df_meta.columns
    labels = np.zeros((n_samples,), dtype=np.float32) if has_labels else None

    for i, raw_idx in enumerate(indices):
        item = raw_data[raw_idx]

        # Verify ID alignment (optional sanity check)
        if item["id"] != ids[i]:
            raise ValueError(
                f"ID Mismatch at index {i}: Meta {ids[i]} vs Raw {item['id']}"
            )

        # Process Bands
        # Raw bands are flattened lists of length 5625
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2

        # Process Angle
        angle_val = item["inc_angle"]
        if angle_val == "na":
            angles[i] = np.nan
        else:
            angles[i] = float(angle_val)

        # Process Label
        if has_labels:
            labels[i] = float(item["is_iceberg"])

    # 3. Save to cache
    save_dict = {"images": images, "angles": angles, "ids": ids}
    if labels is not None:
        save_dict["labels"] = labels

    np.savez_compressed(cache_path, **save_dict)
    # print(f"Data saved to {cache_path}")

    return images, angles, labels, ids


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 2)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            transform (callable, optional): Albumentations transform
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

        # Load global stats for normalization
        # We assume compute_global_stats works as intended in library.utils
        # However, Config already has the constants defined, so we use Config for speed/consistency
        self.b1_min = Config.BAND1_MIN
        self.b1_max = Config.BAND1_MAX
        self.b2_min = Config.BAND2_MIN
        self.b2_max = Config.BAND2_MAX

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Data
        img = self.images[idx]  # (75, 75, 2)
        angle = self.angles[idx]

        # 2. Normalize Bands (Min-Max to [0, 1])
        # Band 1 (HH)
        b1 = (img[:, :, 0] - self.b1_min) / (self.b1_max - self.b1_min)
        # Band 2 (HV)
        b2 = (img[:, :, 1] - self.b2_min) / (self.b2_max - self.b2_min)

        # 3. Create Composite Band (Average of Normalized Bands)
        b3 = (b1 + b2) / 2.0

        # Stack to (75, 75, 3)
        # Albumentations expects HWC
        img_composite = np.dstack((b1, b2, b3)).astype(np.float32)

        # 4. Apply Augmentations / Resizing
        if self.transform:
            augmented = self.transform(image=img_composite)
            img_tensor = augmented["image"]
        else:
            # Fallback to simple tensor conversion (HWC -> CHW)
            img_tensor = torch.from_numpy(img_composite.transpose(2, 0, 1))

        # 5. Normalize Incidence Angle
        # Impute 'na' (NaN) with Mean
        if np.isnan(angle):
            angle = INC_ANGLE_MEAN

        # Standardize
        angle_norm = (angle - INC_ANGLE_MEAN) / INC_ANGLE_STD
        angle_tensor = torch.tensor(angle_norm, dtype=torch.float32)

        # 6. Return
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


def get_dataset(mode, transform=None, load_cached_data=True):
    """
    Factory function to create datasets.

    Args:
        mode (str): 'train', 'val', or 'test'.
        transform (callable): Albumentations transform.
        load_cached_data (bool): Whether to use disk caching.

    Returns:
        IcebergDataset
    """
    if mode == "train":
        metadata_path = Config.TRAIN_META
        json_path = Config.TRAIN_JSON
        cache_name = "train_processed"
    elif mode == "val":
        metadata_path = Config.VAL_META
        json_path = Config.TRAIN_JSON  # Val comes from train.json
        cache_name = "val_processed"
    elif mode == "test":
        metadata_path = Config.TEST_META
        json_path = Config.TEST_JSON
        cache_name = "test_processed"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    images, angles, labels, ids = load_and_process_data(
        metadata_path, json_path, cache_name, load_cached_data=load_cached_data
    )

    dataset = IcebergDataset(
        images=images, angles=angles, labels=labels, transform=transform
    )

    return dataset
