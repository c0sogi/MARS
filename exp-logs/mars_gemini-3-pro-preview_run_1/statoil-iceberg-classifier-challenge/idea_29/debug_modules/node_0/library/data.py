import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles serving of preprocessed images and incidence angles.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and angle
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform is provided
            # Transpose (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Normalize Incidence Angle (Standard Scaling)
        # Formula: (x - mean) / std
        angle = (angle - Config.INC_ANGLE_MEAN) / Config.INC_ANGLE_STD
        angle = torch.tensor(angle, dtype=torch.float32)

        # Return tuple based on availability of labels
        if self.labels is not None:
            # Target shape: (1,) for BCEWithLogitsLoss
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
            return image, angle, label
        else:
            return image, angle


def get_transforms(phase):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train' or 'val'/'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                # ShiftScaleRotate with limit 20 degrees as per idea
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Note: Input is already Min-Max normalized to [0, 1] during preprocessing.
                # We skip standard ImageNet normalization to preserve the SAR signal characteristics
                # as defined in the "Independent Band Normalization" strategy.
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def process_samples(metadata_df, raw_json_data):
    """
    Processes raw JSON data into numpy arrays suitable for training.
    Performs:
    1. Band reshaping (75x75)
    2. Independent Band Normalization (Min-Max)
    3. Composite Band Creation (Avg of B1, B2)
    4. Bicubic Upsampling (224x224)
    5. Angle extraction and imputation
    """
    n_samples = len(metadata_df)

    # Pre-allocate arrays for efficiency
    images = np.zeros(
        (n_samples, Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
    )
    angles = np.zeros(n_samples, dtype=np.float32)

    # Check if labels exist (train/val) or not (test)
    has_labels = "is_iceberg" in metadata_df.columns
    labels = np.zeros(n_samples, dtype=np.float32) if has_labels else None

    print(f"Processing {n_samples} samples...")

    for i, row in metadata_df.iterrows():
        # Get raw data using the sample index mapped in metadata
        idx = row["sample_index"]
        item = raw_json_data[idx]

        # Extract Bands
        # Raw data is flattened 5625 floats
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

        # Independent Band Normalization (Min-Max)
        b1 = (b1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)
        b2 = (b2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

        # Composite Band (Average of Normalized Bands)
        b3 = (b1 + b2) / 2.0

        # Stack to create (75, 75, 3)
        img_75 = np.dstack((b1, b2, b3))

        # Bicubic Upsampling to Target Size (e.g., 224x224)
        img_resized = cv2.resize(
            img_75, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_CUBIC
        )
        images[i] = img_resized

        # Incidence Angle
        # Metadata generation script coerced 'na' to NaN
        ang = row["inc_angle"]
        if pd.isna(ang):
            ang = Config.INC_ANGLE_FILL
        angles[i] = ang

        # Label
        if has_labels:
            labels[i] = row["is_iceberg"]

    return images, angles, labels


def load_dataset(phase, load_cached_data=True):
    """
    Main entry point to load data.
    Handles caching of processed numpy arrays to avoid redundant processing.

    Args:
        phase (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        IcebergDataset: The ready-to-use dataset.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_file = os.path.join(Config.WORK_DIR, f"{phase}_processed.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {phase} data from {cache_file}...")
        data = np.load(cache_file)
        images = data["images"]
        angles = data["angles"]
        # Load labels if they exist in the archive
        labels = data["labels"] if "labels" in data else None

        # Explicitly set labels to None for test set to ensure consistency
        if phase == "test":
            labels = None

    else:
        print(f"Cache miss or reload requested. Processing {phase} data from source...")

        # Identify source files
        if phase == "train":
            meta_path = Config.TRAIN_META
            json_path = Config.TRAIN_JSON
        elif phase == "val":
            meta_path = Config.VAL_META
            json_path = Config.TRAIN_JSON  # Validation set is a subset of train.json
        elif phase == "test":
            meta_path = Config.TEST_META
            json_path = Config.TEST_JSON
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Load Metadata
        df_meta = pd.read_csv(meta_path)

        # Load Raw JSON
        # We load the entire file. Memory is sufficient (220GB).
        print(f"Loading raw JSON from {json_path}...")
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        # Process
        images, angles, labels = process_samples(df_meta, raw_data)

        # Save to cache
        save_dict = {"images": images, "angles": angles}
        if labels is not None:
            save_dict["labels"] = labels

        np.savez(cache_file, **save_dict)
        print(f"Saved processed {phase} data to {cache_file}")

    # Create Dataset
    # Use 'train' transforms for training, 'val' transforms for val/test
    transform_phase = "train" if phase == "train" else "val"
    transform = get_transforms(transform_phase)

    dataset = IcebergDataset(images, angles, labels, transform=transform)
    return dataset
