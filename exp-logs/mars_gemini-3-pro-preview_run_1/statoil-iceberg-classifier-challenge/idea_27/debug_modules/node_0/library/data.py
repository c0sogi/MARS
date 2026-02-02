import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_CACHE_PATH,
    TEST_CACHE_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    RAW_IMG_SIZE,
)
from library.utils import set_seed


def load_and_process_data(load_cached_data=True):
    """
    Loads training and test data from JSON files or cache.
    Reshapes flattened bands to (N, 75, 75).
    Handles 'na' in inc_angle.

    Returns:
        train_data (dict): Contains 'ids', 'band_1', 'band_2', 'inc_angle', 'labels'
        test_data (dict): Contains 'ids', 'band_1', 'band_2', 'inc_angle'
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(TRAIN_CACHE_PATH), exist_ok=True)

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(TRAIN_CACHE_PATH)
        and os.path.exists(TEST_CACHE_PATH)
    ):
        print(f"Loading data from cache: {TRAIN_CACHE_PATH} and {TEST_CACHE_PATH}")
        try:
            train_npz = np.load(TRAIN_CACHE_PATH, allow_pickle=True)
            test_npz = np.load(TEST_CACHE_PATH, allow_pickle=True)

            train_data = {
                "ids": train_npz["ids"],
                "band_1": train_npz["band_1"],
                "band_2": train_npz["band_2"],
                "inc_angle": train_npz["inc_angle"],
                "labels": train_npz["labels"],
            }

            test_data = {
                "ids": test_npz["ids"],
                "band_1": test_npz["band_1"],
                "band_2": test_npz["band_2"],
                "inc_angle": test_npz["inc_angle"],
            }
            return train_data, test_data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from raw JSON.")

    # 2. Process from scratch
    print("Processing raw JSON data...")

    # --- Process Train ---
    with open(TRAIN_JSON, "r") as f:
        raw_train = json.load(f)

    train_ids = []
    train_b1 = []
    train_b2 = []
    train_angle = []
    train_labels = []

    # Collect valid angles to compute mean for imputation
    valid_angles = []

    for item in raw_train:
        train_ids.append(item["id"])
        train_b1.append(np.array(item["band_1"]).reshape(RAW_IMG_SIZE, RAW_IMG_SIZE))
        train_b2.append(np.array(item["band_2"]).reshape(RAW_IMG_SIZE, RAW_IMG_SIZE))
        train_labels.append(item["is_iceberg"])

        angle = item["inc_angle"]
        if angle == "na":
            train_angle.append(np.nan)
        else:
            val = float(angle)
            train_angle.append(val)
            valid_angles.append(val)

    # Impute missing angles with mean
    angle_mean = np.mean(valid_angles)
    train_angle = np.array(
        [angle_mean if np.isnan(x) else x for x in train_angle], dtype=np.float32
    )

    train_data = {
        "ids": np.array(train_ids),
        "band_1": np.array(train_b1, dtype=np.float32),
        "band_2": np.array(train_b2, dtype=np.float32),
        "inc_angle": train_angle,
        "labels": np.array(train_labels, dtype=np.int64),  # Labels are 0 or 1
    }

    # Save Train Cache
    np.savez_compressed(
        TRAIN_CACHE_PATH,
        ids=train_data["ids"],
        band_1=train_data["band_1"],
        band_2=train_data["band_2"],
        inc_angle=train_data["inc_angle"],
        labels=train_data["labels"],
    )

    # --- Process Test ---
    with open(TEST_JSON, "r") as f:
        raw_test = json.load(f)

    test_ids = []
    test_b1 = []
    test_b2 = []
    test_angle = []

    for item in raw_test:
        test_ids.append(item["id"])
        test_b1.append(np.array(item["band_1"]).reshape(RAW_IMG_SIZE, RAW_IMG_SIZE))
        test_b2.append(np.array(item["band_2"]).reshape(RAW_IMG_SIZE, RAW_IMG_SIZE))

        # Test set shouldn't have 'na' usually, but good to be safe.
        # Note: Test set usually doesn't have 'na' per description, but we handle just in case.
        # If 'na' exists in test, we use train mean (angle_mean).
        angle = item["inc_angle"]
        if angle == "na":
            test_angle.append(angle_mean)
        else:
            test_angle.append(float(angle))

    test_data = {
        "ids": np.array(test_ids),
        "band_1": np.array(test_b1, dtype=np.float32),
        "band_2": np.array(test_b2, dtype=np.float32),
        "inc_angle": np.array(test_angle, dtype=np.float32),
    }

    # Save Test Cache
    np.savez_compressed(
        TEST_CACHE_PATH,
        ids=test_data["ids"],
        band_1=test_data["band_1"],
        band_2=test_data["band_2"],
        inc_angle=test_data["inc_angle"],
    )

    print("Data processing complete and cached.")
    return train_data, test_data


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=20,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class IcebergDataset(Dataset):
    def __init__(
        self,
        band_1,
        band_2,
        inc_angle,
        labels=None,
        ids=None,
        variant="A",
        transform=None,
        stats=None,
        angle_stats=None,
    ):
        """
        Args:
            band_1, band_2: Numpy arrays (N, 75, 75)
            inc_angle: Numpy array (N,)
            labels: Numpy array (N,) or None
            ids: Array of IDs
            variant: 'A' (Mean) or 'B' (Difference)
            transform: Albumentations transform
            stats: Dict containing min/max for normalization
            angle_stats: Dict containing mean/std for angle normalization
        """
        self.band_1 = band_1
        self.band_2 = band_2
        self.inc_angle = inc_angle
        self.labels = labels
        self.ids = ids
        self.variant = variant
        self.transform = transform
        self.stats = stats
        self.angle_stats = angle_stats

    def __len__(self):
        return len(self.band_1)

    def __getitem__(self, idx):
        b1 = self.band_1[idx]
        b2 = self.band_2[idx]
        angle = self.inc_angle[idx]

        # 1. Construct 3rd Channel based on Variant
        if self.variant == "A":
            # Variant A: [Band 1, Band 2, Mean]
            b3 = (b1 + b2) / 2.0

            # Normalize
            b1 = (b1 - self.stats["b1_min"]) / (
                self.stats["b1_max"] - self.stats["b1_min"]
            )
            b2 = (b2 - self.stats["b2_min"]) / (
                self.stats["b2_max"] - self.stats["b2_min"]
            )
            b3 = (b3 - self.stats["mean_min"]) / (
                self.stats["mean_max"] - self.stats["mean_min"]
            )

        elif self.variant == "B":
            # Variant B: [Band 1, Band 2, Difference]
            b3 = b1 - b2

            # Normalize
            b1 = (b1 - self.stats["b1_min"]) / (
                self.stats["b1_max"] - self.stats["b1_min"]
            )
            b2 = (b2 - self.stats["b2_min"]) / (
                self.stats["b2_max"] - self.stats["b2_min"]
            )
            b3 = (b3 - self.stats["diff_min"]) / (
                self.stats["diff_max"] - self.stats["diff_min"]
            )

        # Stack channels (H, W, C)
        img = np.dstack((b1, b2, b3))

        # 2. Upsample to 224x224 (Bicubic)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        # 3. Apply Transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]  # Returns Tensor (C, H, W)

        # 4. Normalize Angle
        if self.angle_stats:
            angle = (angle - self.angle_stats["mean"]) / self.angle_stats["std"]

        # Prepare return values
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, angle_tensor, label, self.ids[idx]
        else:
            return img, angle_tensor, self.ids[idx]


def compute_normalization_stats(band_1, band_2):
    """
    Computes global min/max stats for all required channels.
    """
    stats = {}
    stats["b1_min"] = band_1.min()
    stats["b1_max"] = band_1.max()
    stats["b2_min"] = band_2.min()
    stats["b2_max"] = band_2.max()

    mean_band = (band_1 + band_2) / 2.0
    stats["mean_min"] = mean_band.min()
    stats["mean_max"] = mean_band.max()

    diff_band = band_1 - band_2
    stats["diff_min"] = diff_band.min()
    stats["diff_max"] = diff_band.max()

    return stats


def create_dataloaders(fold=0, n_splits=5, variant="A", load_cached_data=True):
    """
    Creates DataLoaders for a specific fold or full training.

    Args:
        fold (int or None): The fold index (0 to n_splits-1). If None, uses all training data.
        n_splits (int): Number of folds for CV.
        variant (str): 'A' or 'B'.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        train_loader, val_loader, test_loader
        (val_loader is None if fold is None)
    """
    set_seed(SEED)

    # 1. Load Data
    train_data, test_data = load_and_process_data(load_cached_data)

    X_b1 = train_data["band_1"]
    X_b2 = train_data["band_2"]
    X_angle = train_data["inc_angle"]
    y = train_data["labels"]
    ids = train_data["ids"]

    # 2. Split Data
    if fold is not None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(skf.split(X_angle, y))
        train_idx, val_idx = splits[fold]

        # Select subsets
        X_b1_train, X_b1_val = X_b1[train_idx], X_b1[val_idx]
        X_b2_train, X_b2_val = X_b2[train_idx], X_b2[val_idx]
        angle_train, angle_val = X_angle[train_idx], X_angle[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        ids_train, ids_val = ids[train_idx], ids[val_idx]

    else:
        # Use full dataset
        X_b1_train = X_b1
        X_b2_train = X_b2
        angle_train = X_angle
        y_train = y
        ids_train = ids

        # Validation is empty/None
        X_b1_val, X_b2_val, angle_val, y_val, ids_val = None, None, None, None, None

    # 3. Compute Stats on Training Subset (Avoid Leakage)
    stats = compute_normalization_stats(X_b1_train, X_b2_train)

    angle_stats = {
        "mean": angle_train.mean(),
        "std": angle_train.std() + 1e-6,  # Avoid div by zero
    }

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        X_b1_train,
        X_b2_train,
        angle_train,
        y_train,
        ids_train,
        variant=variant,
        transform=get_transforms("train"),
        stats=stats,
        angle_stats=angle_stats,
    )

    if fold is not None:
        val_dataset = IcebergDataset(
            X_b1_val,
            X_b2_val,
            angle_val,
            y_val,
            ids_val,
            variant=variant,
            transform=get_transforms("valid"),
            stats=stats,
            angle_stats=angle_stats,
        )
    else:
        val_dataset = None

    # Test Dataset (uses training stats)
    test_dataset = IcebergDataset(
        test_data["band_1"],
        test_data["band_2"],
        test_data["inc_angle"],
        labels=None,
        ids=test_data["ids"],
        variant=variant,
        transform=get_transforms("test"),
        stats=stats,
        angle_stats=angle_stats,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
