import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.utils import (
    set_seed,
    BAND1_MIN,
    BAND1_MAX,
    BAND1_MEAN,
    BAND1_STD,
    BAND2_MIN,
    BAND2_MAX,
    BAND2_MEAN,
    BAND2_STD,
    min_max_scale,
)


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, 75, 75, 3).
            angles (np.ndarray): Array of incidence angles with shape (N,).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (A.Compose, optional): Albumentations transform pipeline.
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

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Fallback if transform doesn't return a tensor (e.g. if ToTensorV2 is missing)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().permute(2, 0, 1)

        # Convert angle to tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=20,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Resize(224, 224, interpolation=cv2.INTER_CUBIC), ToTensorV2()]
        )


def process_and_cache_data(metadata_path, json_path, cache_path, load_cached_data=True):
    """
    Loads raw data based on metadata, processes it (normalization, stacking),
    and caches the result.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path)
            images = data["images"]
            angles = data["angles"]
            labels = data["labels"] if "labels" in data else None
            return images, angles, labels
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data from {json_path} using metadata {metadata_path}...")

    # Load Metadata
    df = pd.read_csv(metadata_path)

    # Load Raw JSON
    # Note: We load the full JSON once. This is memory efficient enough for this dataset size.
    with open(json_path, "r") as f:
        raw_json = json.load(f)

    # Map raw_json by index for O(1) access
    # The metadata 'sample_index' corresponds to the list index in the raw json
    indices = df["sample_index"].values

    processed_images = []
    processed_angles = []
    processed_labels = []

    has_labels = "is_iceberg" in df.columns

    for i, idx in enumerate(indices):
        sample = raw_json[idx]

        # --- Image Processing ---
        # Reshape flattened bands
        b1 = np.array(sample["band_1"]).reshape(75, 75)
        b2 = np.array(sample["band_2"]).reshape(75, 75)

        # Independent Band Normalization (Global Min-Max)
        b1_norm = min_max_scale(b1, BAND1_MIN, BAND1_MAX)
        b2_norm = min_max_scale(b2, BAND2_MIN, BAND2_MAX)

        # Composite Band (Average of Normalized Bands)
        avg_norm = (b1_norm + b2_norm) / 2.0

        # Stack to (75, 75, 3)
        img = np.dstack((b1_norm, b2_norm, avg_norm)).astype(np.float32)
        processed_images.append(img)

        # --- Angle Processing ---
        # We take the angle from the dataframe which handles 'na' as NaN
        angle_val = df.iloc[i]["inc_angle"]
        processed_angles.append(angle_val)

        # --- Label Processing ---
        if has_labels:
            processed_labels.append(df.iloc[i]["is_iceberg"])

    processed_images = np.array(processed_images)
    processed_angles = np.array(processed_angles, dtype=np.float32)

    if has_labels:
        processed_labels = np.array(processed_labels, dtype=np.float32)
    else:
        processed_labels = None

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if processed_labels is not None:
        np.savez(
            cache_path,
            images=processed_images,
            angles=processed_angles,
            labels=processed_labels,
        )
    else:
        np.savez(cache_path, images=processed_images, angles=processed_angles)

    return processed_images, processed_angles, processed_labels


def get_dataloaders(
    input_dir="./input",
    metadata_dir="./metadata",
    cache_dir="./working/idea_30",
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
    debug=False,
):
    """
    Main function to get DataLoaders. Handles imputation and normalization of angles.
    """
    set_seed(42)

    # File Paths
    train_meta = os.path.join(metadata_dir, "train_metadata.csv")
    val_meta = os.path.join(metadata_dir, "val_metadata.csv")
    test_meta = os.path.join(metadata_dir, "test_metadata.csv")

    train_json = os.path.join(input_dir, "train.json")
    test_json = os.path.join(input_dir, "test.json")

    # Process Data
    X_train, a_train, y_train = process_and_cache_data(
        train_meta,
        train_json,
        os.path.join(cache_dir, "train_processed.npz"),
        load_cached_data,
    )
    X_val, a_val, y_val = process_and_cache_data(
        val_meta,
        train_json,
        os.path.join(cache_dir, "val_processed.npz"),
        load_cached_data,
    )
    X_test, a_test, _ = process_and_cache_data(
        test_meta,
        test_json,
        os.path.join(cache_dir, "test_processed.npz"),
        load_cached_data,
    )

    # --- Angle Imputation & Normalization ---
    # 1. Calculate stats from Training set (ignoring NaNs)
    train_angle_mean = np.nanmean(a_train)
    train_angle_std = np.nanstd(a_train)

    print(
        f"Angle Stats (Train) - Mean: {train_angle_mean:.4f}, Std: {train_angle_std:.4f}"
    )

    # 2. Impute NaNs with Train Mean
    a_train = np.where(np.isnan(a_train), train_angle_mean, a_train)
    a_val = np.where(np.isnan(a_val), train_angle_mean, a_val)
    a_test = np.where(np.isnan(a_test), train_angle_mean, a_test)

    # 3. Normalize (Z-score)
    a_train = (a_train - train_angle_mean) / train_angle_std
    a_val = (a_val - train_angle_mean) / train_angle_std
    a_test = (a_test - train_angle_mean) / train_angle_std

    # --- Debug Mode ---
    if debug:
        print("DEBUG MODE: Limiting dataset size to 100 samples.")
        limit = 100
        X_train, a_train, y_train = X_train[:limit], a_train[:limit], y_train[:limit]
        X_val, a_val, y_val = X_val[:limit], a_val[:limit], y_val[:limit]
        X_test, a_test = X_test[:limit], a_test[:limit]

    # --- Create Datasets ---
    train_ds = IcebergDataset(
        X_train, a_train, y_train, transform=get_transforms("train")
    )
    val_ds = IcebergDataset(X_val, a_val, y_val, transform=get_transforms("val"))
    test_ds = IcebergDataset(X_test, a_test, None, transform=get_transforms("test"))

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"DataLoaders created. Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader
