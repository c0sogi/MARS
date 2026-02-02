import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import library.config as config

# =============================================================================
# DATASET CLASS
# =============================================================================


class IcebergDataset(Dataset):
    def __init__(
        self,
        band_1,
        band_2,
        inc_angles,
        labels=None,
        ids=None,
        transform=None,
        stats=None,
    ):
        """
        Args:
            band_1 (np.ndarray): Shape (N, 75, 75)
            band_2 (np.ndarray): Shape (N, 75, 75)
            inc_angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            ids (np.ndarray, optional): Shape (N,)
            transform (albumentations.Compose): Augmentation pipeline.
            stats (dict): Global stats for normalization {'b1_min', 'b1_max', 'b2_min', 'b2_max', 'ang_mean', 'ang_std'}
        """
        self.band_1 = band_1
        self.band_2 = band_2
        self.inc_angles = inc_angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.stats = stats

    def __len__(self):
        return len(self.band_1)

    def __getitem__(self, idx):
        # 1. Retrieve Raw Data
        b1 = self.band_1[idx]
        b2 = self.band_2[idx]
        angle = self.inc_angles[idx]

        # 2. Global Min-Max Normalization
        # Epsilon to avoid div by zero (though unlikely given the data range)
        eps = 1e-6
        b1_norm = (b1 - self.stats["b1_min"]) / (
            self.stats["b1_max"] - self.stats["b1_min"] + eps
        )
        b2_norm = (b2 - self.stats["b2_min"]) / (
            self.stats["b2_max"] - self.stats["b2_min"] + eps
        )

        # 3. Construct Composite Band (Average)
        b3_norm = (b1_norm + b2_norm) / 2.0

        # 4. Stack to create (75, 75, 3)
        # Shape becomes (H, W, C) for Albumentations/OpenCV
        img = np.dstack((b1_norm, b2_norm, b3_norm)).astype(np.float32)

        # 5. Upsample to 224x224 using Bicubic Interpolation
        img = cv2.resize(
            img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC
        )

        # 6. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            # If no transform, just convert to tensor
            # Albumentations ToTensorV2 handles HWC -> CHW
            converter = ToTensorV2()
            img = converter(image=img)["image"]

        # 7. Normalize Incidence Angle (Standard Scaling)
        angle_norm = (angle - self.stats["ang_mean"]) / (self.stats["ang_std"] + eps)
        angle_norm = torch.tensor(angle_norm, dtype=torch.float32)

        # 8. Return
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, angle_norm, label
        else:
            # For test set, return ID as well for submission
            img_id = self.ids[idx]
            return img, angle_norm, img_id


# =============================================================================
# DATA PROCESSING & CACHING
# =============================================================================


def process_json_data(filepath, is_train=True):
    """
    Parses the raw JSON file and extracts bands, angles, and labels.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    ids = []
    band_1 = []
    band_2 = []
    inc_angles = []
    labels = []

    for item in data:
        ids.append(item["id"])
        band_1.append(item["band_1"])
        band_2.append(item["band_2"])

        # Handle incidence angle
        angle = item["inc_angle"]
        if angle == "na":
            inc_angles.append(np.nan)
        else:
            inc_angles.append(float(angle))

        if is_train:
            labels.append(item["is_iceberg"])

    # Convert to numpy arrays
    # Reshape bands to (N, 75, 75)
    band_1 = np.array(band_1, dtype=np.float32).reshape(-1, 75, 75)
    band_2 = np.array(band_2, dtype=np.float32).reshape(-1, 75, 75)
    inc_angles = np.array(inc_angles, dtype=np.float32)
    ids = np.array(ids)

    if is_train:
        labels = np.array(labels, dtype=np.float32)
        return band_1, band_2, inc_angles, labels, ids
    else:
        return band_1, band_2, inc_angles, ids


def load_data(load_cached_data=True):
    """
    Loads data from cache or processes raw JSONs.
    Computes global stats for normalization.
    """
    cache_train = os.path.join(config.CACHE_DIR, "train_processed.npz")
    cache_test = os.path.join(config.CACHE_DIR, "test_processed.npz")

    # Ensure cache dir exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # --- Load Train Data ---
    if load_cached_data and os.path.exists(cache_train):
        print(f"Loading cached train data from {cache_train}")
        train_npz = np.load(cache_train)
        train_b1 = train_npz["band_1"]
        train_b2 = train_npz["band_2"]
        train_angles = train_npz["inc_angles"]
        train_labels = train_npz["labels"]
        train_ids = train_npz["ids"]
    else:
        print("Processing raw train.json...")
        train_b1, train_b2, train_angles, train_labels, train_ids = process_json_data(
            config.TRAIN_JSON, is_train=True
        )
        np.savez(
            cache_train,
            band_1=train_b1,
            band_2=train_b2,
            inc_angles=train_angles,
            labels=train_labels,
            ids=train_ids,
        )

    # --- Load Test Data ---
    if load_cached_data and os.path.exists(cache_test):
        print(f"Loading cached test data from {cache_test}")
        test_npz = np.load(cache_test)
        test_b1 = test_npz["band_1"]
        test_b2 = test_npz["band_2"]
        test_angles = test_npz["inc_angles"]
        test_ids = test_npz["ids"]
    else:
        print("Processing raw test.json...")
        test_b1, test_b2, test_angles, test_ids = process_json_data(
            config.TEST_JSON, is_train=False
        )
        np.savez(
            cache_test,
            band_1=test_b1,
            band_2=test_b2,
            inc_angles=test_angles,
            ids=test_ids,
        )

    # --- Impute Missing Angles in Train ---
    # Calculate mean of valid angles
    valid_angle_mask = ~np.isnan(train_angles)
    angle_mean = np.nanmean(train_angles)

    # Fill NaNs in train
    train_angles = np.where(np.isnan(train_angles), angle_mean, train_angles)

    # Note: Test set usually doesn't have 'na' based on description, but if it did,
    # we would use train mean. Assuming test is clean or numeric.
    # If test has NaNs, we fill with train mean.
    test_angles = np.where(np.isnan(test_angles), angle_mean, test_angles)

    # --- Compute Global Statistics (from Train Set) ---
    stats = {
        "b1_min": np.min(train_b1),
        "b1_max": np.max(train_b1),
        "b2_min": np.min(train_b2),
        "b2_max": np.max(train_b2),
        "ang_mean": angle_mean,
        "ang_std": np.nanstd(train_angles[valid_angle_mask]),
    }

    print("Global Statistics Computed:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    data = {
        "train": (train_b1, train_b2, train_angles, train_labels, train_ids),
        "test": (test_b1, test_b2, test_angles, test_ids),
        "stats": stats,
    }

    return data


# =============================================================================
# AUGMENTATIONS
# =============================================================================


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5
                ),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    elif mode == "test" or mode == "val":
        return A.Compose([ToTensorV2()])
    else:
        raise ValueError(f"Unknown transform mode: {mode}")


# =============================================================================
# DATALOADERS
# =============================================================================


def get_dataloaders(
    fold=0,
    n_folds=5,
    batch_size=config.BATCH_SIZE,
    mode="train_cv",
    load_cached_data=True,
):
    """
    Creates dataloaders for training, validation, or testing.

    Args:
        fold (int): Current fold index (0 to n_folds-1).
        n_folds (int): Number of folds for CV.
        batch_size (int): Batch size.
        mode (str): 'train_cv' (returns train_loader, val_loader),
                    'full_train' (returns loader with all train data),
                    'test' (returns test_loader).
        load_cached_data (bool): Whether to use cached numpy files.
    """

    # Load all data
    data_container = load_data(load_cached_data=load_cached_data)
    train_data = data_container["train"]
    test_data = data_container["test"]
    stats = data_container["stats"]

    train_b1, train_b2, train_angles, train_labels, train_ids = train_data

    if mode == "test":
        test_b1, test_b2, test_angles, test_ids = test_data
        test_dataset = IcebergDataset(
            test_b1,
            test_b2,
            test_angles,
            ids=test_ids,
            transform=get_transforms("test"),
            stats=stats,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        return test_loader

    elif mode == "full_train":
        # Use entire training set
        full_dataset = IcebergDataset(
            train_b1,
            train_b2,
            train_angles,
            labels=train_labels,
            ids=train_ids,
            transform=get_transforms("train"),
            stats=stats,
        )
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    elif mode == "train_cv":
        # Stratified K-Fold Split
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

        # Get indices for the requested fold
        # We iterate to find the specific fold indices
        for i, (train_idx, val_idx) in enumerate(skf.split(train_b1, train_labels)):
            if i == fold:
                break

        # Create Train Subset
        train_dataset = IcebergDataset(
            train_b1[train_idx],
            train_b2[train_idx],
            train_angles[train_idx],
            labels=train_labels[train_idx],
            ids=train_ids[train_idx],
            transform=get_transforms("train"),
            stats=stats,
        )

        # Create Val Subset
        val_dataset = IcebergDataset(
            train_b1[val_idx],
            train_b2[val_idx],
            train_angles[val_idx],
            labels=train_labels[val_idx],
            ids=train_ids[val_idx],
            transform=get_transforms("val"),
            stats=stats,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    else:
        raise ValueError(f"Unknown mode: {mode}")
