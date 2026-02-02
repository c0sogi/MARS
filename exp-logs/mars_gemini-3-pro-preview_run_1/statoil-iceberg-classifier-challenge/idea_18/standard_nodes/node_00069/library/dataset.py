import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 224, 224, 3), float32.
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), int/float.
            ids (np.ndarray, optional): Shape (N,), string (for test set).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in pipeline)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Prepare return dictionary or tuple based on availability of labels/ids
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, torch.tensor(angle, dtype=torch.float32), label
        elif self.ids is not None:
            return image, torch.tensor(angle, dtype=torch.float32), self.ids[idx]
        else:
            return image, torch.tensor(angle, dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        phase (str): 'train' for augmentations, 'val'/'test' for deterministic.
    """
    if phase == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.0,
                    rotate_limit=Config.ROTATION_DEGREES,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                # Convert to Tensor (HWC -> CHW)
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def process_samples(metadata_df, raw_data_dict):
    """
    Process raw JSON data into numpy arrays based on metadata indices.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'sample_index', 'inc_angle', etc.
        raw_data_dict (dict/list): The loaded JSON content (list of dicts).

    Returns:
        images (np.ndarray): (N, 224, 224, 3)
        angles (np.ndarray): (N,)
        labels (np.ndarray): (N,) or None
        ids (np.ndarray): (N,)
    """
    num_samples = len(metadata_df)

    # Pre-allocate arrays
    images = np.zeros(
        (num_samples, Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
    )
    angles = np.zeros(num_samples, dtype=np.float32)
    ids = np.empty(num_samples, dtype=object)

    has_labels = "is_iceberg" in metadata_df.columns
    labels = np.zeros(num_samples, dtype=np.float32) if has_labels else None

    # Extract indices to access raw data efficiently
    indices = metadata_df["sample_index"].values

    # Metadata values
    meta_angles = metadata_df["inc_angle"].values
    meta_ids = metadata_df["id"].values
    if has_labels:
        meta_labels = metadata_df["is_iceberg"].values

    for i, raw_idx in enumerate(indices):
        item = raw_data_dict[raw_idx]

        # 1. Image Processing
        # Reshape to 75x75
        band_1 = np.array(item["band_1"]).reshape(75, 75)
        band_2 = np.array(item["band_2"]).reshape(75, 75)

        # Global Min-Max Normalization
        b1_norm = (band_1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)
        b2_norm = (band_2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

        # Composite Band (Average)
        b3_norm = (b1_norm + b2_norm) / 2.0

        # Stack
        img_75 = np.dstack((b1_norm, b2_norm, b3_norm))

        # Bicubic Upsampling to 224x224
        img_224 = cv2.resize(
            img_75, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_CUBIC
        )
        images[i] = img_224

        # 2. Angle Processing
        # Use value from metadata (which already coerced 'na' to NaN)
        angle_val = meta_angles[i]

        # Fill NaN with Mean
        if np.isnan(angle_val):
            angle_val = Config.INC_ANGLE_MEAN

        # Standard Scaling
        angle_norm = (angle_val - Config.INC_ANGLE_MEAN) / Config.INC_ANGLE_STD
        angles[i] = angle_norm

        # 3. ID
        ids[i] = meta_ids[i]

        # 4. Label
        if has_labels:
            labels[i] = meta_labels[i]

    return images, angles, labels, ids


def load_data(metadata_path, json_path, cache_prefix, load_cached_data=True):
    """
    Loads data with caching mechanism.

    Args:
        metadata_path (str): Path to metadata CSV.
        json_path (str): Path to raw JSON.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, angles, labels, ids)
    """
    # Define cache paths
    cache_img_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    cache_ang_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_angles.npy")
    cache_lbl_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_img_path)
        and os.path.exists(cache_ang_path)
        and os.path.exists(cache_ids_path)
    )
    # Check label cache only if it's not test set (heuristic based on prefix)
    is_test = "test" in cache_prefix
    if not is_test:
        cache_exists = cache_exists and os.path.exists(cache_lbl_path)

    if load_cached_data and cache_exists:
        print(f"Loading {cache_prefix} data from cache...")
        images = np.load(cache_img_path)
        angles = np.load(cache_ang_path)
        ids = np.load(cache_ids_path, allow_pickle=True)
        labels = np.load(cache_lbl_path) if not is_test else None
        return images, angles, labels, ids

    # Process from scratch
    print(f"Processing {cache_prefix} data from raw files...")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)

    # Load raw JSON
    # Optimization: Only load the necessary JSON file
    # Note: train.json and test.json are loaded entirely into memory.
    # Given the dataset size (1600 train, 8000 test), this is feasible (approx 200MB-1GB raw text).
    print(f"Loading raw JSON: {json_path}")
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    images, angles, labels, ids = process_samples(df_meta, raw_data)

    # Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(cache_img_path, images)
    np.save(cache_ang_path, angles)
    np.save(cache_ids_path, ids)
    if labels is not None:
        np.save(cache_lbl_path, labels)

    return images, angles, labels, ids
