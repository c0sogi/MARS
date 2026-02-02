import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library import config

# =============================================================================
# CACHING & RAW DATA PROCESSING
# =============================================================================


def process_json_data(json_path, cache_prefix, load_cached_data=True):
    """
    Parses the JSON file and converts it to numpy arrays.
    Implements caching to .npy files in the working directory.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    angle_path = os.path.join(cache_dir, f"{cache_prefix}_angles.npy")
    id_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")
    label_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")

    # Check if cache exists and we want to load it
    if load_cached_data:
        has_labels = os.path.exists(label_path)
        if (
            os.path.exists(img_path)
            and os.path.exists(angle_path)
            and os.path.exists(id_path)
        ):
            # If it's the training set, we also need labels
            if "train" in cache_prefix and not has_labels:
                pass  # Cache incomplete
            else:
                print(f"Loading cached {cache_prefix} data from {cache_dir}...")
                images = np.load(img_path)
                angles = np.load(angle_path)
                ids = np.load(id_path)
                labels = np.load(label_path) if has_labels else None
                return images, angles, labels, ids

    print(f"Processing raw data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Pre-allocate lists
    images_list = []
    angles_list = []
    ids_list = []
    labels_list = []

    has_labels_in_json = len(data) > 0 and "is_iceberg" in data[0]

    for item in data:
        # Process Bands
        # Each band is a list of 5625 floats. Reshape to (75, 75)
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        # Stack to (75, 75, 2)
        img = np.dstack((b1, b2))
        images_list.append(img)

        # Process Angle
        angle = item["inc_angle"]
        if angle == "na":
            angles_list.append(np.nan)
        else:
            angles_list.append(float(angle))

        # Process ID
        ids_list.append(item["id"])

        # Process Label
        if has_labels_in_json:
            labels_list.append(item["is_iceberg"])

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.float32)
    angles = np.array(angles_list, dtype=np.float32)
    ids = np.array(ids_list)
    labels = np.array(labels_list, dtype=np.float32) if has_labels_in_json else None

    # Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(img_path, images)
    np.save(angle_path, angles)
    np.save(id_path, ids)
    if labels is not None:
        np.save(label_path, labels)

    return images, angles, labels, ids


# =============================================================================
# AUGMENTATIONS
# =============================================================================


def get_transforms(mode="train"):
    """
    Returns the Albumentations composition for training or inference.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(
                    limit=config.ROTATION_LIMIT,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Test/Val: Just convert to tensor (resizing is handled in Dataset)
        return A.Compose([ToTensorV2()])


# =============================================================================
# DATASET CLASS
# =============================================================================


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels, ids, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 2)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray): Shape (N,) or None
            ids (np.ndarray): Shape (N,)
            transform (albumentations.Compose): Augmentation pipeline
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

        # Impute missing angles with global mean (approx 39.28 from analysis)
        # We do this once during init
        self.angles = np.where(np.isnan(self.angles), 39.28, self.angles)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Data
        # Shape: (75, 75, 2)
        img = self.images[idx]
        angle = self.angles[idx]
        id_ = self.ids[idx]

        # 2. Global Min-Max Normalization
        # Band 1 (HH)
        b1 = img[:, :, 0]
        b1 = (b1 - config.BAND_1_MIN) / (config.BAND_1_MAX - config.BAND_1_MIN)

        # Band 2 (HV)
        b2 = img[:, :, 1]
        b2 = (b2 - config.BAND_2_MIN) / (config.BAND_2_MAX - config.BAND_2_MIN)

        # 3. Create Composite Band (Average)
        b3 = (b1 + b2) / 2.0

        # 4. Stack to (75, 75, 3)
        # Values are now approx [0, 1]
        img_composite = np.dstack((b1, b2, b3))

        # 5. Upsample to (224, 224) using Bicubic Interpolation
        img_resized = cv2.resize(
            img_composite,
            (config.IMG_SIZE, config.IMG_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        # 6. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            # Fallback if no transform provided
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1))

        # 7. Prepare Label and Angle
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor, id_
        else:
            return img_tensor, angle_tensor, id_


# =============================================================================
# DATALOADERS
# =============================================================================


def get_dataloaders(load_cached_data=True, full_train=False):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.
        full_train (bool): If True, combines Train and Val sets for final training.

    Returns:
        train_loader, val_loader, test_loader
        (val_loader is None if full_train is True)
    """

    # 1. Load Metadata
    df_train_meta = pd.read_csv(config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(config.VAL_META_PATH)
    df_test_meta = pd.read_csv(config.TEST_META_PATH)

    # 2. Load/Process Raw Data (Cached)
    # Train/Val data comes from train.json
    all_train_imgs, all_train_angles, all_train_labels, all_train_ids = (
        process_json_data(config.TRAIN_JSON, "train", load_cached_data)
    )

    # Test data comes from test.json
    test_imgs, test_angles, _, test_ids = process_json_data(
        config.TEST_JSON, "test", load_cached_data
    )

    # 3. Select Subsets based on Metadata Indices
    # The metadata contains 'sample_index' which maps to the index in the raw arrays

    # Train Subset
    train_indices = df_train_meta["sample_index"].values
    X_train = all_train_imgs[train_indices]
    a_train = all_train_angles[train_indices]
    y_train = all_train_labels[train_indices]
    id_train = all_train_ids[train_indices]

    # Val Subset
    val_indices = df_val_meta["sample_index"].values
    X_val = all_train_imgs[val_indices]
    a_val = all_train_angles[val_indices]
    y_val = all_train_labels[val_indices]
    id_val = all_train_ids[val_indices]

    # Test Subset (Metadata order might differ from json order, so we reorder)
    test_indices = df_test_meta["sample_index"].values
    X_test = test_imgs[test_indices]
    a_test = test_angles[test_indices]
    id_test = test_ids[test_indices]

    # 4. Handle Full Train Mode
    if full_train:
        # Concatenate Train and Val
        X_train = np.concatenate([X_train, X_val], axis=0)
        a_train = np.concatenate([a_train, a_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        id_train = np.concatenate([id_train, id_val], axis=0)

        # Val becomes None
        val_loader = None

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train, a_train, y_train, id_train, transform=get_transforms(mode="train")
    )

    if not full_train:
        val_dataset = IcebergDataset(
            X_val, a_val, y_val, id_val, transform=get_transforms(mode="val")
        )

    test_dataset = IcebergDataset(
        X_test, a_test, None, id_test, transform=get_transforms(mode="test")
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    if not full_train:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
