import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg Detection.
    Handles 3-channel image construction and augmentation.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 224, 224, 3), float32, scaled [0, 1].
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), float32.
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default to converting to tensor if no transform provided
            # Albumentations ToTensorV2 handles HWC -> CHW
            converter = ToTensorV2()
            image = converter(image=image)["image"]

        # Convert angle to tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def _process_images(bands_1, bands_2, image_size=224):
    """
    Processes raw bands into resized, scaled, 3-channel images.

    Channels:
    0: Band 1 (scaled)
    1: Band 2 (scaled)
    2: Mean of Ch0 and Ch1
    """
    n_samples = len(bands_1)
    processed_images = np.zeros(
        (n_samples, image_size, image_size, 3), dtype=np.float32
    )

    # Global Stats
    b1_min = Config.GLOBAL_STATS["band_1"]["min"]
    b1_max = Config.GLOBAL_STATS["band_1"]["max"]
    b2_min = Config.GLOBAL_STATS["band_2"]["min"]
    b2_max = Config.GLOBAL_STATS["band_2"]["max"]

    for i in range(n_samples):
        # Reshape to 75x75
        b1 = np.array(bands_1[i]).reshape(75, 75).astype(np.float32)
        b2 = np.array(bands_2[i]).reshape(75, 75).astype(np.float32)

        # Global Min-Max Scaling
        b1 = (b1 - b1_min) / (b1_max - b1_min)
        b2 = (b2 - b2_min) / (b2_max - b2_min)

        # Resize to target size using Bicubic interpolation
        b1_resized = cv2.resize(
            b1, (image_size, image_size), interpolation=cv2.INTER_CUBIC
        )
        b2_resized = cv2.resize(
            b2, (image_size, image_size), interpolation=cv2.INTER_CUBIC
        )

        # Construct 3rd channel (Mean)
        b3_resized = (b1_resized + b2_resized) / 2.0

        # Stack
        processed_images[i, :, :, 0] = b1_resized
        processed_images[i, :, :, 1] = b2_resized
        processed_images[i, :, :, 2] = b3_resized

    return processed_images


def load_data(load_cached_data=True):
    """
    Loads data from JSON or Cache.
    Handles imputation and preprocessing.

    Returns:
        train_images, train_angles, train_labels, test_images, test_angles, test_ids
    """
    # Cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "train_img": os.path.join(cache_dir, "train_images.npy"),
        "train_ang": os.path.join(cache_dir, "train_angles.npy"),
        "train_lbl": os.path.join(cache_dir, "train_labels.npy"),
        "test_img": os.path.join(cache_dir, "test_images.npy"),
        "test_ang": os.path.join(cache_dir, "test_angles.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check cache
    if load_cached_data:
        if all(os.path.exists(p) for p in paths.values()):
            print("Loading data from cache...")
            train_images = np.load(paths["train_img"])
            train_angles = np.load(paths["train_ang"])
            train_labels = np.load(paths["train_lbl"])
            test_images = np.load(paths["test_img"])
            test_angles = np.load(paths["test_ang"])
            test_ids = np.load(paths["test_ids"])
            return (
                train_images,
                train_angles,
                train_labels,
                test_images,
                test_angles,
                test_ids,
            )
        else:
            print("Cache missing or incomplete. Processing from scratch...")

    # Load Raw Data
    print("Loading raw JSON files...")
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # --- Process Train ---
    print("Processing training data...")
    train_ids = [item["id"] for item in train_data]
    train_labels = np.array(
        [item["is_iceberg"] for item in train_data], dtype=np.float32
    )

    # Handle Angles (Impute 'na' with mean)
    train_angles_raw = [item["inc_angle"] for item in train_data]
    valid_angles = [a for a in train_angles_raw if a != "na"]
    angle_mean = np.mean(valid_angles)

    train_angles = np.array(
        [angle_mean if a == "na" else a for a in train_angles_raw], dtype=np.float32
    )

    # Process Images
    train_b1 = [item["band_1"] for item in train_data]
    train_b2 = [item["band_2"] for item in train_data]
    train_images = _process_images(train_b1, train_b2, Config.IMAGE_SIZE)

    # --- Process Test ---
    print("Processing test data...")
    test_ids = np.array([item["id"] for item in test_data])

    # Handle Angles (Use train mean if 'na' exists, though description says unlikely)
    test_angles_raw = [item["inc_angle"] for item in test_data]
    test_angles = np.array(
        [angle_mean if a == "na" else a for a in test_angles_raw], dtype=np.float32
    )

    # Process Images
    test_b1 = [item["band_1"] for item in test_data]
    test_b2 = [item["band_2"] for item in test_data]
    test_images = _process_images(test_b1, test_b2, Config.IMAGE_SIZE)

    # Save to Cache
    print("Saving processed data to cache...")
    np.save(paths["train_img"], train_images)
    np.save(paths["train_ang"], train_angles)
    np.save(paths["train_lbl"], train_labels)
    np.save(paths["test_img"], test_images)
    np.save(paths["test_ang"], test_angles)
    np.save(paths["test_ids"], test_ids)

    return train_images, train_angles, train_labels, test_images, test_angles, test_ids


def get_dataloaders(
    train_idx, val_idx, batch_size=Config.BATCH_SIZE, load_cached_data=True
):
    """
    Creates DataLoaders for a specific bag (split).

    Args:
        train_idx (list/array): Indices for the training set (Bag).
        val_idx (list/array): Indices for the validation set (OOB).
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    train_images, train_angles, train_labels, _, _, _ = load_data(load_cached_data)

    # Subset
    X_train = train_images[train_idx]
    a_train = train_angles[train_idx]
    y_train = train_labels[train_idx]

    X_val = train_images[val_idx]
    a_val = train_angles[val_idx]
    y_val = train_labels[val_idx]

    # Define Transforms
    # Training: Horizontal Flip, Vertical Flip, Rotation
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            ToTensorV2(),
        ]
    )

    # Validation: No augmentation, just Tensor conversion
    val_transform = A.Compose([ToTensorV2()])

    # Create Datasets
    train_dataset = IcebergDataset(X_train, a_train, y_train, transform=train_transform)
    val_dataset = IcebergDataset(X_val, a_val, y_val, transform=val_transform)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoader for the test set.

    Returns:
        test_loader, test_ids
    """
    _, _, _, test_images, test_angles, test_ids = load_data(load_cached_data)

    # Transform: Just Tensor conversion
    test_transform = A.Compose([ToTensorV2()])

    test_dataset = IcebergDataset(
        test_images, test_angles, labels=None, transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_ids
