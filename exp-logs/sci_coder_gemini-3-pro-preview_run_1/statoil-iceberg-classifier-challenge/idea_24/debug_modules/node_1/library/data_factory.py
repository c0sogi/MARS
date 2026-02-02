import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config
from library import utils

# ==========================================
# Constants & Statistics
# ==========================================
# Statistics derived from data analysis
ANGLE_MEAN = 39.2829
ANGLE_STD = 3.8362


# ==========================================
# Dataset Class
# ==========================================
class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg Classification.
    Handles 3-channel construction, Bicubic upsampling, and metadata fusion.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, 75, 75, 2).
            angles (np.ndarray): Array of shape (N,).
            labels (np.ndarray, optional): Array of shape (N,).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve raw data
        # shape: (75, 75, 2)
        img = self.images[idx]
        angle = self.angles[idx]

        # 2. Independent Band Normalization (Global Min-Max)
        # Band 1: HH
        b1 = (img[:, :, 0] - config.BAND1_MIN) / (config.BAND1_MAX - config.BAND1_MIN)
        # Band 2: HV
        b2 = (img[:, :, 1] - config.BAND2_MIN) / (config.BAND2_MAX - config.BAND2_MIN)

        # 3. Construct Composite Band (Average of Normalized Bands)
        b3 = (b1 + b2) / 2.0

        # 4. Stack to form 3-channel image (75, 75, 3)
        img_composite = np.dstack((b1, b2, b3))

        # 5. Bicubic Upsampling to 224x224
        # We resize before augmentation to ensure high-quality geometric transformations
        img_resized = cv2.resize(
            img_composite,
            (config.IMG_SIZE, config.IMG_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        # 6. Apply Augmentations (Albumentations)
        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            # Fallback to ToTensorV2 equivalent if no transform provided
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()

        # 7. Normalize Incidence Angle (Standard Scaling)
        angle_norm = (angle - ANGLE_MEAN) / ANGLE_STD
        angle_tensor = torch.tensor(angle_norm, dtype=torch.float32)

        # 8. Return Data
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # Return dummy label for test set
            return img_tensor, angle_tensor, torch.tensor(-1.0)


# ==========================================
# Transforms
# ==========================================
def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for deterministic formatting.
    """
    if mode == "train":
        return A.Compose(
            [
                # Discrete Rotation
                A.RandomRotate90(p=0.5),
                # Continuous Geometric Augmentation (Limit +/- 20 degrees)
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                # Flips
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Conversion
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ==========================================
# Data Processing & Loading
# ==========================================
def load_processed_data(load_cached_data=True):
    """
    Loads processed data from cache or processes raw JSON files.

    Returns:
        tuple: (train_images, train_angles, train_labels, train_ids,
                test_images, test_angles, test_ids)
    """
    cache_train_path = os.path.join(config.CACHE_DIR, "train_processed.npz")
    cache_test_path = os.path.join(config.CACHE_DIR, "test_processed.npz")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_train_path)
        and os.path.exists(cache_test_path)
    ):
        print(f"Loading cached data from {config.CACHE_DIR}...")
        train_data = np.load(cache_train_path)
        test_data = np.load(cache_test_path)

        return (
            train_data["images"],
            train_data["angles"],
            train_data["labels"],
            train_data["ids"],
            test_data["images"],
            test_data["angles"],
            test_data["ids"],
        )

    print("Cache not found or ignored. Processing raw JSON data...")

    # 2. Process Training Data
    print(f"Loading {config.TRAIN_JSON}...")
    with open(config.TRAIN_JSON, "r") as f:
        raw_train = json.load(f)

    train_images = []
    train_angles = []
    train_labels = []
    train_ids = []

    # Calculate mean angle for imputation from valid training data
    valid_angles = [
        float(item["inc_angle"]) for item in raw_train if item["inc_angle"] != "na"
    ]
    impute_angle = np.mean(valid_angles)
    print(f"Imputing missing angles with mean: {impute_angle:.4f}")

    for item in raw_train:
        # Extract Bands
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        # Stack to (75, 75, 2)
        img = np.dstack((b1, b2))
        train_images.append(img)

        # Extract Angle (Impute if 'na')
        if item["inc_angle"] == "na":
            train_angles.append(impute_angle)
        else:
            train_angles.append(float(item["inc_angle"]))

        # Extract Label and ID
        train_labels.append(item["is_iceberg"])
        train_ids.append(item["id"])

    # Convert to numpy arrays
    train_images = np.array(train_images, dtype=np.float32)
    train_angles = np.array(train_angles, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    train_ids = np.array(train_ids)

    # 3. Process Test Data
    print(f"Loading {config.TEST_JSON}...")
    with open(config.TEST_JSON, "r") as f:
        raw_test = json.load(f)

    test_images = []
    test_angles = []
    test_ids = []

    for item in raw_test:
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        img = np.dstack((b1, b2))
        test_images.append(img)

        # Extract Angle
        try:
            ang = float(item["inc_angle"])
        except (ValueError, TypeError):
            ang = impute_angle
        test_angles.append(ang)

        test_ids.append(item["id"])

    test_images = np.array(test_images, dtype=np.float32)
    test_angles = np.array(test_angles, dtype=np.float32)
    test_ids = np.array(test_ids)

    # 4. Save to Cache
    print(f"Saving processed data to {config.CACHE_DIR}...")
    np.savez(
        cache_train_path,
        images=train_images,
        angles=train_angles,
        labels=train_labels,
        ids=train_ids,
    )
    np.savez(cache_test_path, images=test_images, angles=test_angles, ids=test_ids)

    return (
        train_images,
        train_angles,
        train_labels,
        train_ids,
        test_images,
        test_angles,
        test_ids,
    )


def get_dataloaders(
    batch_size=config.BATCH_SIZE, load_cached_data=True, train_idxs=None, val_idxs=None
):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached data.
        train_idxs (array-like, optional): Indices for training set. If None, uses metadata split.
        val_idxs (array-like, optional): Indices for validation set. If None, uses metadata split.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load all data
    t_imgs, t_angs, t_lbls, t_ids, test_imgs, test_angs, test_ids = load_processed_data(
        load_cached_data
    )

    # Determine Splits
    if train_idxs is None or val_idxs is None:
        print("Using default split from metadata...")
        df_train_meta = pd.read_csv(config.TRAIN_META_PATH)
        df_val_meta = pd.read_csv(config.VAL_META_PATH)

        train_idxs = df_train_meta["sample_index"].values
        val_idxs = df_val_meta["sample_index"].values
    else:
        print(f"Using provided indices. Train: {len(train_idxs)}, Val: {len(val_idxs)}")

    # Create Datasets
    train_ds = IcebergDataset(
        t_imgs[train_idxs],
        t_angs[train_idxs],
        t_lbls[train_idxs],
        transform=get_transforms("train"),
    )

    val_ds = IcebergDataset(
        t_imgs[val_idxs],
        t_angs[val_idxs],
        t_lbls[val_idxs],
        transform=get_transforms("val"),
    )

    test_ds = IcebergDataset(test_imgs, test_angs, transform=get_transforms("test"))

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Batch Norm stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_all_data(load_cached_data=True):
    """
    Helper function to retrieve all processed data arrays.
    Useful for Cross-Validation splitting in the main script.
    """
    return load_processed_data(load_cached_data)
