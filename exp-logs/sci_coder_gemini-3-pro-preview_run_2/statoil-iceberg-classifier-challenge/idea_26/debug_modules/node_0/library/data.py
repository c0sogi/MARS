import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from library import config, utils


def load_data(load_cached_data=True):
    """
    Loads and processes the raw image data from JSON files.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary mapping 'id' to numpy image array of shape (3, 75, 75).
    """
    cache_path = os.path.join(config.CACHE_DIR, "processed_data.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                # np.load returns a NpzFile object, we need to extract the dict
                # We saved it as 'images_dict' containing the dictionary
                images_dict = data["images_dict"].item()
            print(f"Successfully loaded {len(images_dict)} images from cache.")
            return images_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from raw data...")

    # 2. Process from Scratch
    print("Processing raw JSON data...")

    # Load Train Data
    print(f"Loading {config.TRAIN_JSON}...")
    with open(config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    # Load Test Data
    print(f"Loading {config.TEST_JSON}...")
    with open(config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Combine lists
    all_data = train_data + test_data
    images_dict = {}

    print(f"Processing {len(all_data)} images...")
    for item in all_data:
        img_id = item["id"]

        # Extract bands and reshape to 75x75
        band_1 = np.array(item["band_1"]).reshape(config.IMG_HEIGHT, config.IMG_WIDTH)
        band_2 = np.array(item["band_2"]).reshape(config.IMG_HEIGHT, config.IMG_WIDTH)

        # Calculate Band 3 (Average)
        band_3 = (band_1 + band_2) / 2.0

        # Stack to create (3, 75, 75) image
        # Channels: [HH, HV, Avg]
        img = np.stack([band_1, band_2, band_3], axis=0).astype(np.float32)

        images_dict[img_id] = img

    # 3. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    try:
        np.savez_compressed(cache_path, images_dict=images_dict)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return images_dict


def get_transforms(mode="train"):
    """
    Returns the Albumentations augmentation pipeline.

    Args:
        mode (str): "train" or "val"/"test".

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        # Rotational Invariance: 0, 90, 180, 270 degrees + Horizontal Flip
        # RandomRotate90(p=1.0) picks randomly from [0, 90, 180, 270]
        # HorizontalFlip(p=0.5) adds reflection symmetry
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5 if config.AUGMENT_HFLIP else 0.0),
                A.RandomRotate90(p=1.0 if config.AUGMENT_ROTATION else 0.0),
            ]
        )
    else:
        # Identity transform for validation and test
        return A.Compose([])


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles metadata alignment, image retrieval, scaling, and augmentation.
    """

    def __init__(
        self, metadata, images_dict, scaler=None, transform=None, inc_angle_fill=None
    ):
        """
        Args:
            metadata (pd.DataFrame or str): DataFrame or path to CSV containing metadata.
            images_dict (dict): Dictionary mapping IDs to image arrays (3, 75, 75).
            scaler (FoldScaler, optional): Fitted scaler for normalization.
            transform (A.Compose, optional): Albumentations transforms.
            inc_angle_fill (float, optional): Value to fill NaN incidence angles.
                                              If None, calculates mean from the provided metadata.
        """
        # Load metadata if path provided
        if isinstance(metadata, str):
            self.meta = pd.read_csv(metadata)
        else:
            self.meta = metadata.copy()

        self.images_dict = images_dict
        self.scaler = scaler
        self.transform = transform

        # --- Handle Incidence Angle ---
        # Convert to numeric, coercing errors to NaN
        self.meta["inc_angle"] = pd.to_numeric(self.meta["inc_angle"], errors="coerce")

        # Determine fill value for NaNs
        if inc_angle_fill is not None:
            self.fill_value = inc_angle_fill
        else:
            # Calculate mean from current metadata (ignoring NaNs)
            self.fill_value = self.meta["inc_angle"].mean()
            # Fallback for edge case where all are NaN
            if np.isnan(self.fill_value):
                self.fill_value = 0.0

        # Fill NaNs
        self.meta["inc_angle"] = self.meta["inc_angle"].fillna(self.fill_value)

        # --- Prepare Arrays for Fast Access ---
        self.ids = self.meta["id"].values
        self.inc_angles = self.meta["inc_angle"].values.astype(np.float32)

        # Handle Labels (only exist for train/val)
        if "is_iceberg" in self.meta.columns:
            self.labels = self.meta["is_iceberg"].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Retrieve image: Shape (3, 75, 75)
        img = self.images_dict[img_id]

        # Retrieve incidence angle
        inc_angle = self.inc_angles[idx]

        # --- Augmentation ---
        if self.transform:
            # Albumentations expects HWC (Height, Width, Channels)
            # Transpose (3, 75, 75) -> (75, 75, 3)
            img_hwc = np.transpose(img, (1, 2, 0))

            # Apply transforms
            augmented = self.transform(image=img_hwc)
            img_aug = augmented["image"]

            # Transpose back to CHW -> (3, 75, 75)
            img = np.transpose(img_aug, (2, 0, 1))

        # --- Scaling ---
        if self.scaler:
            # FoldScaler expects (N, C, H, W)
            # Add batch dimension -> (1, 3, 75, 75)
            img_batch = img[np.newaxis, ...]

            # Transform
            img_scaled_batch = self.scaler.transform(img_batch)

            # Remove batch dimension
            img = img_scaled_batch[0]

        # --- Tensor Conversion ---
        img_tensor = torch.from_numpy(img)
        inc_angle_tensor = torch.tensor(inc_angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, inc_angle_tensor, label_tensor
        else:
            return img_tensor, inc_angle_tensor
