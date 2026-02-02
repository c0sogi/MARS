import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from library import config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(
                    config.IMAGE_SIZE, config.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC
                ),
                A.HorizontalFlip(p=0.5 if config.DO_HORIZONTAL_FLIP else 0.0),
                A.VerticalFlip(p=0.5 if config.DO_VERTICAL_FLIP else 0.0),
                A.Rotate(limit=config.ROTATION_RANGE, p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(
                    config.IMAGE_SIZE, config.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC
                ),
                ToTensorV2(),
            ]
        )


class IcebergDataset(Dataset):
    def __init__(self, images, angles, ids, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 3), float32, range [0, 1].
            angles (np.ndarray): Shape (N,), float32, normalized.
            ids (np.ndarray): Shape (N,), string IDs.
            labels (np.ndarray, optional): Shape (N,), int/float labels.
            transform (A.Compose, optional): Albumentations pipeline.
        """
        self.images = images
        self.angles = angles
        self.ids = ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert angle to tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            # For inference, return ID to track predictions
            id_val = self.ids[idx]
            return image, angle, id_val


def _process_data(metadata_df, json_path):
    """
    Reads JSON data based on metadata indices, processes bands, and extracts fields.
    """
    # Load the full JSON file
    # Note: This might be memory intensive but fits within 220GB RAM.
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Pre-allocate arrays
    n_samples = len(metadata_df)
    images = np.zeros(
        (n_samples, config.ORIGINAL_SIZE, config.ORIGINAL_SIZE, 3), dtype=np.float32
    )
    angles = np.zeros(n_samples, dtype=np.float32)
    ids = np.empty(n_samples, dtype=object)
    labels = (
        np.zeros(n_samples, dtype=np.float32)
        if "is_iceberg" in metadata_df.columns
        else None
    )

    # Get indices from metadata to access raw list correctly
    indices = metadata_df["sample_index"].values

    for i, sample_idx in enumerate(indices):
        item = raw_data[sample_idx]

        # 1. Process Bands
        # Flattened list of 5625 floats
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(
            config.ORIGINAL_SIZE, config.ORIGINAL_SIZE
        )
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(
            config.ORIGINAL_SIZE, config.ORIGINAL_SIZE
        )

        # Normalize Independent Bands
        b1 = (b1 - config.BAND1_MIN) / (config.BAND1_MAX - config.BAND1_MIN)
        b2 = (b2 - config.BAND2_MIN) / (config.BAND2_MAX - config.BAND2_MIN)

        # Create Composite Band (Mean)
        b3 = (b1 + b2) / 2.0

        # Stack: (75, 75, 3)
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2
        images[i, :, :, 2] = b3

        # 2. Process Angle
        # Handle 'na' by converting to None/NaN first, handled later by bulk imputation
        ang = item["inc_angle"]
        if ang == "na":
            angles[i] = np.nan
        else:
            angles[i] = float(ang)

        # 3. ID
        ids[i] = item["id"]

        # 4. Label
        if labels is not None:
            labels[i] = item["is_iceberg"]

    return images, angles, ids, labels


def _get_angle_stats():
    """
    Computes angle mean and std from the training metadata for imputation and normalization.
    """
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    # Convert 'na' to NaN
    angles = pd.to_numeric(df_train["inc_angle"], errors="coerce")

    mu = angles.mean()
    std = angles.std()
    return mu, std


def load_dataset(mode="train", load_cached_data=True):
    """
    Main function to load the dataset. Handles caching and preprocessing.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        IcebergDataset: The ready-to-use dataset.
    """
    # Determine paths
    if mode == "train":
        meta_path = config.TRAIN_META_PATH
        json_path = config.TRAIN_JSON
    elif mode == "val":
        meta_path = config.VAL_META_PATH
        json_path = config.TRAIN_JSON
    elif mode == "test":
        meta_path = config.TEST_META_PATH
        json_path = config.TEST_JSON
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Cache filenames
    cache_dir = config.WORKING_DIR  # Ensure this is idea_13
    os.makedirs(cache_dir, exist_ok=True)

    prefix = f"{mode}"
    img_cache = os.path.join(cache_dir, f"{prefix}_images.npy")
    ang_cache = os.path.join(cache_dir, f"{prefix}_angles.npy")
    id_cache = os.path.join(cache_dir, f"{prefix}_ids.npy")
    lbl_cache = os.path.join(cache_dir, f"{prefix}_labels.npy")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(img_cache)
        and os.path.exists(ang_cache)
        and os.path.exists(id_cache)
    ):
        # Check label cache existence if not test
        if mode == "test" or os.path.exists(lbl_cache):
            print(f"[{mode.upper()}] Loading cached data from {cache_dir}...")
            images = np.load(img_cache)
            angles = np.load(ang_cache)
            ids = np.load(id_cache, allow_pickle=True)
            labels = np.load(lbl_cache) if mode != "test" else None

            # Create Dataset
            transform = get_transforms(mode)
            return IcebergDataset(images, angles, ids, labels, transform)

    # 2. Process from Scratch
    print(f"[{mode.upper()}] Processing data from scratch...")
    df_meta = pd.read_csv(meta_path)
    images, angles, ids, labels = _process_data(df_meta, json_path)

    # 3. Impute and Normalize Angles
    # We use training set statistics for consistency across all splits
    mu, std = _get_angle_stats()

    # Impute NaNs with mean
    nan_mask = np.isnan(angles)
    angles[nan_mask] = mu

    # Normalize (Z-score)
    angles = (angles - mu) / std

    # 4. Save to Cache
    print(f"[{mode.upper()}] Saving processed data to cache...")
    np.save(img_cache, images)
    np.save(ang_cache, angles)
    np.save(id_cache, ids)
    if labels is not None:
        np.save(lbl_cache, labels)

    # 5. Create Dataset
    transform = get_transforms(mode)
    return IcebergDataset(images, angles, ids, labels, transform)
