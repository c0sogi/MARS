import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from library.config import Config
from library.utils import seed_everything


# -------------------------------------------------------------------------
# Augmentation Pipeline
# -------------------------------------------------------------------------
def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for resizing only.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(
                    height=Config.IMG_SIZE,
                    width=Config.IMG_SIZE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                ToTensorV2(),
            ]
        )


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------
class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        PyTorch Dataset for Iceberg vs Ship classification.

        Args:
            images (np.ndarray): Shape (N, 75, 75, 3), float32, range [0, 1].
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), float32 (0 or 1).
            ids (np.ndarray, optional): Shape (N,), string IDs.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

        # Incidence Angle Statistics for Normalization (Standard Scaling)
        # Derived from training set analysis: Mean ~39.28, Std ~3.84
        self.angle_mean = 39.28
        self.angle_std = 3.84

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Image Processing
        image = self.images[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W)
        else:
            # Fallback if no transform provided (shouldn't happen in pipeline)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # 2. Angle Processing
        angle = self.angles[idx]
        # Impute NaNs with mean
        if np.isnan(angle):
            angle = self.angle_mean
        # Standard Scale
        angle = (angle - self.angle_mean) / self.angle_std
        angle = torch.tensor(angle, dtype=torch.float32)

        # 3. Return Tuple
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            # For test set, return ID for submission mapping
            img_id = self.ids[idx] if self.ids is not None else ""
            return image, angle, img_id


# -------------------------------------------------------------------------
# Data Loading & Processing Logic
# -------------------------------------------------------------------------
def _process_data(metadata_path, json_path, data_type, save_dir):
    """
    Internal function to load JSON, process bands, and save processed arrays.
    """
    print(f"Processing {data_type} data from {json_path}...")

    # Load Metadata
    df_meta = pd.read_csv(metadata_path)

    # Load Raw JSON
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Map metadata rows to raw data using sample_index
    indices = df_meta["sample_index"].values
    samples = [raw_data[i] for i in indices]

    # --- Process Images ---
    # Reshape flattened bands to 75x75
    b1_list = [np.array(s["band_1"]).reshape(75, 75) for s in samples]
    b2_list = [np.array(s["band_2"]).reshape(75, 75) for s in samples]

    b1 = np.array(b1_list, dtype=np.float32)
    b2 = np.array(b2_list, dtype=np.float32)

    # Normalize (Min-Max)
    b1 = (b1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)
    b2 = (b2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

    # Create Composite Band (Average)
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 75, 75, 3)
    images = np.stack([b1, b2, b3], axis=-1)

    # --- Process Angles ---
    # Use the 'inc_angle' from metadata which handles 'na' -> NaN coercion
    angles = df_meta["inc_angle"].values.astype(np.float32)

    # --- Process IDs ---
    ids = df_meta["id"].values

    # --- Process Labels (if train/val) ---
    labels = None
    if "is_iceberg" in df_meta.columns:
        labels = df_meta["is_iceberg"].values.astype(np.float32)

    # --- Save to Cache ---
    np.save(os.path.join(save_dir, f"{data_type}_images.npy"), images)
    np.save(os.path.join(save_dir, f"{data_type}_angles.npy"), angles)
    np.save(os.path.join(save_dir, f"{data_type}_ids.npy"), ids)
    if labels is not None:
        np.save(os.path.join(save_dir, f"{data_type}_labels.npy"), labels)

    return images, angles, labels, ids


def load_dataset_data(data_type, load_cached_data=True):
    """
    Loads dataset arrays (images, angles, labels, ids).
    Uses caching to avoid re-processing raw JSON.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, angles, labels, ids)
    """
    save_dir = Config.WORKING_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Define file paths
    img_path = os.path.join(save_dir, f"{data_type}_images.npy")
    ang_path = os.path.join(save_dir, f"{data_type}_angles.npy")
    lbl_path = os.path.join(save_dir, f"{data_type}_labels.npy")
    id_path = os.path.join(save_dir, f"{data_type}_ids.npy")

    # Check cache
    has_cache = (
        os.path.exists(img_path)
        and os.path.exists(ang_path)
        and os.path.exists(id_path)
    )
    if data_type in ["train", "val"]:
        has_cache = has_cache and os.path.exists(lbl_path)

    if load_cached_data and has_cache:
        # print(f"Loading {data_type} data from cache...")
        images = np.load(img_path)
        angles = np.load(ang_path)
        ids = np.load(id_path, allow_pickle=True)  # IDs are strings
        labels = np.load(lbl_path) if data_type in ["train", "val"] else None
        return images, angles, labels, ids

    # If no cache or forced reload, process from scratch
    if data_type == "train":
        return _process_data(
            Config.TRAIN_META_PATH, Config.TRAIN_JSON, "train", save_dir
        )
    elif data_type == "val":
        return _process_data(Config.VAL_META_PATH, Config.TRAIN_JSON, "val", save_dir)
    elif data_type == "test":
        return _process_data(Config.TEST_META_PATH, Config.TEST_JSON, "test", save_dir)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def get_dataloaders(load_cached_data=True, extra_data=None):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Supports Semi-Supervised Learning by accepting extra pseudo-labeled data.

    Args:
        load_cached_data (bool): Use cached .npy files if available.
        extra_data (tuple, optional): (images, angles, labels) tuple of pseudo-labeled data
                                      to append to the training set.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.SEED)

    # 1. Load Base Data
    train_imgs, train_angs, train_lbls, train_ids = load_dataset_data(
        "train", load_cached_data
    )
    val_imgs, val_angs, val_lbls, val_ids = load_dataset_data("val", load_cached_data)
    test_imgs, test_angs, _, test_ids = load_dataset_data("test", load_cached_data)

    # 2. Create Datasets
    train_ds = IcebergDataset(
        train_imgs,
        train_angs,
        train_lbls,
        train_ids,
        transform=get_transforms(mode="train"),
    )

    val_ds = IcebergDataset(
        val_imgs, val_angs, val_lbls, val_ids, transform=get_transforms(mode="val")
    )

    test_ds = IcebergDataset(
        test_imgs,
        test_angs,
        labels=None,
        ids=test_ids,
        transform=get_transforms(mode="test"),
    )

    # 3. Handle Semi-Supervised Learning (Extra Data)
    if extra_data is not None:
        ex_imgs, ex_angs, ex_lbls = extra_data
        print(f"Augmenting training set with {len(ex_imgs)} pseudo-labeled samples.")

        # Create dataset for extra data (apply same train transforms)
        extra_ds = IcebergDataset(
            ex_imgs, ex_angs, ex_lbls, ids=None, transform=get_transforms(mode="train")
        )

        # Concatenate
        train_ds = ConcatDataset([train_ds, extra_ds])

    # 4. Create Loaders
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

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
