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
from library.utils import set_seed

# =============================================================================
# Data Caching and Processing
# =============================================================================


def process_and_cache_data(data_type="train", load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches it.

    Args:
        data_type (str): "train" (includes val) or "test".
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        dict: A dictionary containing 'images', 'angles', 'ids', and 'labels' (if train).
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{data_type}_processed.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {data_type} data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            result = {
                "images": data["images"],
                "angles": data["angles"],
                "ids": data["ids"],
            }
            if "labels" in data:
                result["labels"] = data["labels"]
            return result
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {data_type} data from raw JSON...")

    if data_type == "train":
        # Load both train and val metadata to reconstruct full labeled set
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_meta = pd.concat([df_train, df_val], ignore_index=True)
        json_path = Config.TRAIN_JSON
    else:
        df_meta = pd.read_csv(Config.TEST_META_PATH)
        json_path = Config.TEST_JSON

    # Load Raw JSON
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Map raw data by ID for O(1) access or use index mapping if consistent
    # The metadata contains 'sample_index' which maps directly to the list index
    # We filter/reorder based on metadata to ensure alignment

    indices = df_meta["sample_index"].values
    ids = df_meta["id"].values

    # Pre-allocate arrays
    n_samples = len(indices)
    images = np.zeros((n_samples, 75, 75, 2), dtype=np.float32)
    angles = np.zeros(n_samples, dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.float32) if data_type == "train" else None

    for i, idx in enumerate(indices):
        item = raw_data[idx]

        # Verify ID alignment
        if item["id"] != ids[i]:
            raise ValueError(
                f"ID Mismatch at index {i}: Meta {ids[i]} vs JSON {item['id']}"
            )

        # Extract Bands
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2

        # Extract Angle
        angle = item["inc_angle"]
        if angle == "na" or angle is None:
            angles[i] = Config.INC_ANGLE_FILL
        else:
            angles[i] = float(angle)

        # Extract Label
        if data_type == "train":
            labels[i] = item["is_iceberg"]

    # 3. Save to Cache
    print(f"Saving processed {data_type} data to {cache_file}...")
    save_dict = {"images": images, "angles": angles, "ids": ids}
    if labels is not None:
        save_dict["labels"] = labels

    np.savez_compressed(cache_file, **save_dict)

    return {"images": images, "angles": angles, "ids": ids, "labels": labels}


# =============================================================================
# Dataset Class
# =============================================================================


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 2)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            transform (albumentations.Compose): Augmentation pipeline
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Raw Data
        # Shape: (75, 75, 2)
        img_raw = self.images[idx]
        angle = self.angles[idx]

        # 2. Independent Band Normalization
        # Band 1 (HH)
        b1 = (img_raw[:, :, 0] - Config.BAND1_MIN) / (
            Config.BAND1_MAX - Config.BAND1_MIN
        )
        # Band 2 (HV)
        b2 = (img_raw[:, :, 1] - Config.BAND2_MIN) / (
            Config.BAND2_MAX - Config.BAND2_MIN
        )

        # 3. Construct Composite Band (Average of Normalized B1 and B2)
        b3 = (b1 + b2) / 2.0

        # Stack to (75, 75, 3)
        img_stacked = np.dstack((b1, b2, b3)).astype(np.float32)

        # 4. Bicubic Upsampling
        # Resize from 75x75 to 224x224
        img_resized = cv2.resize(
            img_stacked,
            (Config.IMG_SIZE, Config.IMG_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        # 5. Augmentations
        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            # Fallback if no transform provided (should usually be ToTensorV2)
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1))

        # 6. Normalize Incidence Angle
        angle_norm = (angle - Config.INC_ANGLE_MEAN) / Config.INC_ANGLE_STD
        angle_tensor = torch.tensor([angle_norm], dtype=torch.float32)

        # 7. Return
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


# =============================================================================
# Transforms
# =============================================================================


def get_transforms(phase="train"):
    """
    Returns Albumentations transform pipeline.

    Args:
        phase (str): 'train' or 'test'/'val'
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Closure over cardinal axes
                A.RandomRotate90(p=0.5),
                # Smooth decision boundary
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# =============================================================================
# Data Loaders
# =============================================================================


def get_kfold_loaders(fold_idx, n_splits=5, load_cached_data=True):
    """
    Generates DataLoaders for Phase 1 (Calibration/CV).
    Uses StratifiedKFold on the full labeled dataset.
    """
    # Load full labeled data
    data = process_and_cache_data("train", load_cached_data)
    images = data["images"]
    angles = data["angles"]
    labels = data["labels"]

    # Stratified Split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
    splits = list(skf.split(images, labels))

    train_idx, val_idx = splits[fold_idx]

    # Create Datasets
    train_ds = IcebergDataset(
        images[train_idx],
        angles[train_idx],
        labels[train_idx],
        transform=get_transforms("train"),
    )

    val_ds = IcebergDataset(
        images[val_idx],
        angles[val_idx],
        labels[val_idx],
        transform=get_transforms("val"),
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_production_loader(load_cached_data=True):
    """
    Generates a DataLoader for Phase 2 (Production).
    Uses 100% of the labeled data for training.
    """
    data = process_and_cache_data("train", load_cached_data)

    ds = IcebergDataset(
        data["images"],
        data["angles"],
        data["labels"],
        transform=get_transforms("train"),
    )

    loader = DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def get_test_loader(load_cached_data=True):
    """
    Generates a DataLoader for Inference.
    """
    data = process_and_cache_data("test", load_cached_data)

    ds = IcebergDataset(
        data["images"], data["angles"], labels=None, transform=get_transforms("test")
    )

    loader = DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader, data["ids"]
