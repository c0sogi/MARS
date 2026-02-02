import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config

# -------------------------------------------------------------------------
# Data Processing Logic
# -------------------------------------------------------------------------


def process_bands(band_1, band_2):
    """
    Applies Corrected Composite Fusion:
    1. Global Min-Max Norm on Band 1 using Config stats.
    2. Global Min-Max Norm on Band 2 using Config stats.
    3. Band 3 = Mean(Norm_B1, Norm_B2).
    4. Stack to create (75, 75, 3) image.
    """
    # Reshape flattened lists to 75x75 arrays
    b1 = np.array(band_1).reshape(75, 75)
    b2 = np.array(band_2).reshape(75, 75)

    # Normalize Band 1 (HH)
    b1_norm = (b1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)

    # Normalize Band 2 (HV)
    b2_norm = (b2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

    # Compute Composite Band 3 (Average of normalized bands)
    b3_norm = (b1_norm + b2_norm) / 2.0

    # Stack channels: (75, 75, 3)
    # The resulting values are generally in [0, 1]
    img = np.dstack((b1_norm, b2_norm, b3_norm))
    return img.astype(np.float32)


def load_data(load_cached_data=True):
    """
    Loads raw JSON data and Metadata, processes bands and angles, and returns numpy arrays.
    Implements disk-based caching to avoid re-processing on every run.

    Returns:
        train_data (dict): keys ['images', 'angles', 'labels', 'ids']
        test_data (dict): keys ['images', 'angles', 'ids']
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    train_imgs_path = os.path.join(cache_dir, "train_images.npy")
    train_angles_path = os.path.join(cache_dir, "train_angles.npy")
    train_labels_path = os.path.join(cache_dir, "train_labels.npy")
    train_ids_path = os.path.join(cache_dir, "train_ids.npy")

    test_imgs_path = os.path.join(cache_dir, "test_images.npy")
    test_angles_path = os.path.join(cache_dir, "test_angles.npy")
    test_ids_path = os.path.join(cache_dir, "test_ids.npy")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_imgs_path)
        and os.path.exists(train_angles_path)
        and os.path.exists(train_labels_path)
        and os.path.exists(train_ids_path)
        and os.path.exists(test_imgs_path)
        and os.path.exists(test_angles_path)
        and os.path.exists(test_ids_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading processed data from cache at {cache_dir}...")
        train_images = np.load(train_imgs_path)
        train_angles = np.load(train_angles_path)
        train_labels = np.load(train_labels_path)
        train_ids = np.load(train_ids_path, allow_pickle=True)

        test_images = np.load(test_imgs_path)
        test_angles = np.load(test_angles_path)
        test_ids = np.load(test_ids_path, allow_pickle=True)

    else:
        print("Cache not found or ignored. Processing data from scratch...")

        # Load Metadata
        # Combine train and val metadata to treat as a single training pool (100% data)
        df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
        df_val_meta = pd.read_csv(Config.VAL_META_PATH)
        df_train_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

        df_test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Load Raw JSON Data
        print("Loading raw JSON files...")
        with open(Config.TRAIN_JSON, "r") as f:
            raw_train_data = json.load(f)
        with open(Config.TEST_JSON, "r") as f:
            raw_test_data = json.load(f)

        # --- Pre-calculate Angle Statistics for Normalization ---
        # We use all valid angles in the training set to compute mean/std
        valid_angles = []
        for item in raw_train_data:
            val = item.get("inc_angle")
            try:
                val_float = float(val)
                if not np.isnan(val_float):
                    valid_angles.append(val_float)
            except (ValueError, TypeError):
                continue

        angle_mean = np.mean(valid_angles)
        angle_std = np.std(valid_angles)
        print(f"Angle Statistics - Mean: {angle_mean:.4f}, Std: {angle_std:.4f}")

        # --- Process Training Data ---
        print("Processing training samples...")
        train_images = []
        train_angles = []
        train_labels = []
        train_ids = []

        for _, row in df_train_full.iterrows():
            idx = row["sample_index"]
            item = raw_train_data[idx]

            # Process Image
            img = process_bands(item["band_1"], item["band_2"])
            train_images.append(img)

            # Process Angle (Impute -> Normalize)
            try:
                ang = float(item["inc_angle"])
                if np.isnan(ang):
                    ang = angle_mean
            except (ValueError, TypeError):
                ang = angle_mean

            ang_norm = (ang - angle_mean) / angle_std
            train_angles.append(ang_norm)

            # Label & ID
            train_labels.append(item["is_iceberg"])
            train_ids.append(item["id"])

        train_images = np.array(train_images)
        train_angles = np.array(train_angles, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)
        train_ids = np.array(train_ids)

        # --- Process Test Data ---
        print("Processing test samples...")
        test_images = []
        test_angles = []
        test_ids = []

        for _, row in df_test_meta.iterrows():
            idx = row["sample_index"]
            item = raw_test_data[idx]

            # Process Image
            img = process_bands(item["band_1"], item["band_2"])
            test_images.append(img)

            # Process Angle
            try:
                ang = float(item["inc_angle"])
                if np.isnan(ang):
                    ang = angle_mean
            except (ValueError, TypeError):
                ang = angle_mean

            ang_norm = (ang - angle_mean) / angle_std
            test_angles.append(ang_norm)

            # ID
            test_ids.append(item["id"])

        test_images = np.array(test_images)
        test_angles = np.array(test_angles, dtype=np.float32)
        test_ids = np.array(test_ids)

        # --- Save to Cache ---
        print("Saving processed data to cache...")
        np.save(train_imgs_path, train_images)
        np.save(train_angles_path, train_angles)
        np.save(train_labels_path, train_labels)
        np.save(train_ids_path, train_ids)

        np.save(test_imgs_path, test_images)
        np.save(test_angles_path, test_angles)
        np.save(test_ids_path, test_ids)

    return (
        {
            "images": train_images,
            "angles": train_angles,
            "labels": train_labels,
            "ids": train_ids,
        },
        {"images": test_images, "angles": test_angles, "ids": test_ids},
    )


# -------------------------------------------------------------------------
# Augmentation & Transforms
# -------------------------------------------------------------------------


def get_transforms(phase="train"):
    """
    Constructs the Albumentations transform pipeline.

    Args:
        phase (str): 'train' for augmentation, 'val' for resizing/norm only.
    """
    if phase == "train":
        return A.Compose(
            [
                # Upsample to 224x224 using Bicubic Interpolation
                A.Resize(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=0.0,
                    rotate_limit=Config.ROTATION_LIMIT,  # +/- 20 degrees
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                # Normalize to ImageNet statistics (required for Pretrained ResNet)
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / TTA
        return A.Compose(
            [
                A.Resize(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (75, 75, 3)
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply Transforms (Augmentation + Resize + Norm)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert angle to tensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle_tensor, label
        else:
            return image, angle_tensor


# -------------------------------------------------------------------------
# DataLoader Factories
# -------------------------------------------------------------------------


def get_dataloaders(fold_index=0, full_fit=False, load_cached_data=True):
    """
    Creates DataLoaders for Training and Validation.

    Args:
        fold_index (int): Index of the fold for Cross-Validation (0 to N_FOLDS-1).
        full_fit (bool): If True, trains on 100% of the data (Phase 2).
        load_cached_data (bool): Whether to attempt loading from disk cache.

    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set (None if full_fit=True).
    """
    # Load all training data
    train_data, _ = load_data(load_cached_data=load_cached_data)

    X = train_data["images"]
    y = train_data["labels"]
    angles = train_data["angles"]

    # Get transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    if full_fit:
        print("Initializing Full-Fit DataLoader (100% Data)...")
        # Use entire dataset for training
        train_dataset = IcebergDataset(X, angles, y, transform=train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        return train_loader, None

    else:
        print(f"Initializing Fold {fold_index} DataLoaders...")
        # Stratified K-Fold Split
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )
        splits = list(skf.split(X, y))

        train_idx, val_idx = splits[fold_index]

        # Create Subsets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        ang_train, ang_val = angles[train_idx], angles[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train, ang_train, y_train, transform=train_transform
        )
        val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=val_transform)

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the Test set.

    Returns:
        test_loader: DataLoader for inference.
        test_ids: Array of image IDs corresponding to the loader order.
    """
    _, test_data = load_data(load_cached_data=load_cached_data)

    X_test = test_data["images"]
    angles_test = test_data["angles"]
    ids_test = test_data["ids"]

    # Use validation transform (Resize + Norm, no flips/rotations)
    test_transform = get_transforms("val")

    test_dataset = IcebergDataset(
        X_test, angles_test, labels=None, transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids_test
