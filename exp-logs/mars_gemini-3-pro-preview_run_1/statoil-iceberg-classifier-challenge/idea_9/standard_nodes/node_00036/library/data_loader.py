import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    Uses Bicubic interpolation for better upsampling quality.
    Cite solution_lesson_node_00035
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(
                    Config.IMAGE_SIZE, Config.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Continuous rotation as specified in the idea
                A.Rotate(limit=Config.ROTATION_DEGREES, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(
                    Config.IMAGE_SIZE, Config.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC
                ),
                ToTensorV2(),
            ]
        )


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Custom Dataset for Iceberg/Ship classification.

        Args:
            images (np.ndarray): Shape (N, 75, 75, 3), normalized float32.
            angles (np.ndarray): Shape (N,), incidence angles.
            labels (np.ndarray, optional): Shape (N,), binary targets.
            transform (A.Compose, optional): Albumentations transforms.
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

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Prepare output dictionary
        sample = {
            "image": image,  # Tensor [C, H, W]
            "angle": torch.tensor(angle, dtype=torch.float32),
        }

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sample


def process_json_data(json_path, is_train=True):
    """
    Parses the raw JSON file and extracts bands, angles, and labels.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ids = []
    band_1 = []
    band_2 = []
    angles = []
    labels = []

    for item in data:
        ids.append(item["id"])
        band_1.append(item["band_1"])
        band_2.append(item["band_2"])
        angles.append(item["inc_angle"])
        if is_train:
            labels.append(item["is_iceberg"])

    # Convert to numpy
    # Reshape bands to (N, 75, 75)
    b1 = np.array(band_1, dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(band_2, dtype=np.float32).reshape(-1, 75, 75)

    # Create 3rd channel: Mean of Band 1 and Band 2
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 75, 75, 3)
    images = np.stack([b1, b2, b3], axis=-1)

    # Process angles
    # Replace 'na' with NaN, then we will impute later
    angles_processed = []
    for a in angles:
        try:
            angles_processed.append(float(a))
        except ValueError:
            angles_processed.append(np.nan)
    angles = np.array(angles_processed, dtype=np.float32)

    ids = np.array(ids)

    if is_train:
        labels = np.array(labels, dtype=np.float32)
        return ids, images, angles, labels
    else:
        return ids, images, angles, None


def load_and_process_data(load_cached_data=True):
    """
    Loads data from JSON, processes it (normalization, imputation), and caches it.

    Returns:
        tuple: (X_train, ang_train, y_train, train_ids, X_test, ang_test, test_ids)
    """
    Config.create_directories()

    # Cache file paths
    cache_files = {
        "train_img": os.path.join(Config.WORKING_DIR, "train_images.npy"),
        "train_ang": os.path.join(Config.WORKING_DIR, "train_angles.npy"),
        "train_lbl": os.path.join(Config.WORKING_DIR, "train_labels.npy"),
        "train_ids": os.path.join(Config.WORKING_DIR, "train_ids.npy"),
        "test_img": os.path.join(Config.WORKING_DIR, "test_images.npy"),
        "test_ang": os.path.join(Config.WORKING_DIR, "test_angles.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from working directory...")
        X_train = np.load(cache_files["train_img"])
        ang_train = np.load(cache_files["train_ang"])
        y_train = np.load(cache_files["train_lbl"])
        ids_train = np.load(cache_files["train_ids"], allow_pickle=True)

        X_test = np.load(cache_files["test_img"])
        ang_test = np.load(cache_files["test_ang"])
        ids_test = np.load(cache_files["test_ids"], allow_pickle=True)

        return X_train, ang_train, y_train, ids_train, X_test, ang_test, ids_test

    print("Processing data from scratch...")

    # 1. Load Raw Data
    train_ids, train_imgs, train_angles, train_labels = process_json_data(
        Config.TRAIN_JSON, is_train=True
    )
    test_ids, test_imgs, test_angles, _ = process_json_data(
        Config.TEST_JSON, is_train=False
    )

    # 2. Impute Angles
    # Calculate mean from valid training angles
    angle_mean = np.nanmean(train_angles)

    # Fill NaNs in Train and Test
    train_angles = np.where(np.isnan(train_angles), angle_mean, train_angles)
    test_angles = np.where(np.isnan(test_angles), angle_mean, test_angles)

    # 3. Global Min-Max Normalization
    # Compute stats on Training set only to prevent leakage
    # We normalize each channel independently
    # train_imgs shape: (N, 75, 75, 3)

    for i in range(3):
        channel_data = train_imgs[:, :, :, i]
        _min = channel_data.min()
        _max = channel_data.max()

        # Avoid division by zero
        denom = _max - _min
        if denom == 0:
            denom = 1.0

        # Apply to Train
        train_imgs[:, :, :, i] = (train_imgs[:, :, :, i] - _min) / denom

        # Apply to Test
        test_imgs[:, :, :, i] = (test_imgs[:, :, :, i] - _min) / denom

    # 4. Save to Cache
    np.save(cache_files["train_img"], train_imgs)
    np.save(cache_files["train_ang"], train_angles)
    np.save(cache_files["train_lbl"], train_labels)
    np.save(cache_files["train_ids"], train_ids)

    np.save(cache_files["test_img"], test_imgs)
    np.save(cache_files["test_ang"], test_angles)
    np.save(cache_files["test_ids"], test_ids)

    print("Data processed and cached.")
    return (
        train_imgs,
        train_angles,
        train_labels,
        train_ids,
        test_imgs,
        test_angles,
        test_ids,
    )


def get_loaders(fold_idx=0, stage="teacher", load_cached_data=True):
    """
    Creates DataLoaders for training and inference.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1). Used for Teacher stage.
        stage (str): 'teacher' (K-Fold) or 'student' (Full Data) or 'test'.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        dict: Contains 'train_loader', 'val_loader' (if applicable), 'test_loader'.
    """
    seed_everything(Config.SEED)

    # Load Data
    X_train, ang_train, y_train, ids_train, X_test, ang_test, ids_test = (
        load_and_process_data(load_cached_data)
    )

    loaders = {}

    # --- Test Loader (Always needed for submission) ---
    test_ds = IcebergDataset(
        X_test, ang_test, labels=None, transform=get_transforms(mode="test")
    )
    loaders["test_loader"] = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )
    loaders["test_ids"] = ids_test

    if stage == "test":
        return loaders

    # --- Training Loaders ---

    if stage == "student":
        # Full Data Training
        # We might have soft targets from the Teacher stage
        soft_targets = None
        if os.path.exists(Config.OOF_PREDICTIONS_PATH):
            print(f"Loading Soft Targets from {Config.OOF_PREDICTIONS_PATH}")
            soft_targets = np.load(Config.OOF_PREDICTIONS_PATH)
            # Ensure alignment: OOF predictions should be in same order as loaded data
            # Since we cache data and OOF is generated from that cached data order, it matches.
        else:
            print(
                "Warning: No soft targets found for Student stage. Training with hard labels only."
            )

        train_ds = IcebergDataset(
            X_train,
            ang_train,
            labels=y_train,
            soft_targets=soft_targets,
            transform=get_transforms(mode="train"),
        )

        loaders["train_loader"] = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=True,
        )
        # No validation loader for Student (trained on all data),
        # or we could use the test set as dummy val if needed, but usually we just train for fixed epochs.

    elif stage == "teacher":
        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Get indices for the requested fold
        # X_train is just used for length, y_train for stratification
        fold_generator = skf.split(X_train, y_train)

        # Iterate to find the specific fold
        train_idx, val_idx = next(
            x for i, x in enumerate(fold_generator) if i == fold_idx
        )

        # Subset data
        X_tr, ang_tr, y_tr = (
            X_train[train_idx],
            ang_train[train_idx],
            y_train[train_idx],
        )
        X_val, ang_val, y_val = X_train[val_idx], ang_train[val_idx], y_train[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_tr, ang_tr, labels=y_tr, transform=get_transforms(mode="train")
        )
        val_ds = IcebergDataset(
            X_val, ang_val, labels=y_val, transform=get_transforms(mode="val")
        )

        # Create Loaders
        loaders["train_loader"] = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=True,
        )

        loaders["val_loader"] = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Store validation indices to help reconstruct OOF later if needed
        loaders["val_indices"] = val_idx

    return loaders
