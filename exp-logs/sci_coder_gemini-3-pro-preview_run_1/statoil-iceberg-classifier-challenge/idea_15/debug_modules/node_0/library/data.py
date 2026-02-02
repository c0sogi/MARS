import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from library.config import Config


def load_and_process_json(json_path, cache_prefix, load_cached_data=True):
    """
    Loads JSON data, extracts fields, handles 'na' in inc_angle, and caches as .npy.
    Returns a dict of numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    files = {
        "ids": os.path.join(cache_dir, f"{cache_prefix}_ids.npy"),
        "images": os.path.join(cache_dir, f"{cache_prefix}_images.npy"),
        "angles": os.path.join(cache_dir, f"{cache_prefix}_angles.npy"),
        "labels": os.path.join(cache_dir, f"{cache_prefix}_labels.npy"),
    }

    # Check if cache exists and we want to load it
    if load_cached_data:
        all_exist = True
        for k, v in files.items():
            # Labels only exist for training data
            if k == "labels" and "test" in cache_prefix:
                continue
            if not os.path.exists(v):
                all_exist = False
                break

        if all_exist:
            data = {}
            data["ids"] = np.load(files["ids"], allow_pickle=True)
            data["images"] = np.load(files["images"])
            data["angles"] = np.load(files["angles"])
            if "train" in cache_prefix:
                data["labels"] = np.load(files["labels"])
            return data

    # Process from scratch
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    ids = []
    band_1 = []
    band_2 = []
    angles = []
    labels = []

    is_train = "train" in cache_prefix

    for item in raw_data:
        ids.append(item["id"])
        band_1.append(item["band_1"])
        band_2.append(item["band_2"])

        angle = item["inc_angle"]
        if angle == "na":
            angles.append(np.nan)
        else:
            angles.append(float(angle))

        if is_train:
            labels.append(item["is_iceberg"])

    # Convert to numpy
    ids = np.array(ids)

    # Reshape images: Flattened 5625 -> (N, 75, 75)
    b1 = np.array(band_1, dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(band_2, dtype=np.float32).reshape(-1, 75, 75)

    # Stack to (N, 75, 75, 2)
    images = np.stack([b1, b2], axis=-1)

    angles = np.array(angles, dtype=np.float32)

    # Handle Missing Angles
    # Task description states 'na' only exists in training data.
    # We impute with the mean of valid angles in this set.
    if is_train:
        valid_mask = ~np.isnan(angles)
        mean_angle = np.mean(angles[valid_mask])
        angles[np.isnan(angles)] = mean_angle
        labels = np.array(labels, dtype=np.float32)
    else:
        # For test set, if there were NaNs (unlikely per desc), we would fill with global mean
        # But per description, we expect valid angles.
        # Just in case, fill with analysis mean ~39.28
        if np.isnan(angles).any():
            angles[np.isnan(angles)] = 39.28

    # Save to cache
    np.save(files["ids"], ids)
    np.save(files["images"], images)
    np.save(files["angles"], angles)
    if is_train:
        np.save(files["labels"], labels)

    data = {"ids": ids, "images": images, "angles": angles}
    if is_train:
        data["labels"] = labels

    return data


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data: (75, 75, 2)
        img = self.images[idx]
        angle = self.angles[idx]

        # 1. Independent Band Normalization (Global Min-Max)
        # Band 1 (HH)
        b1 = img[:, :, 0]
        b1 = (b1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)

        # Band 2 (HV)
        b2 = img[:, :, 1]
        b2 = (b2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

        # 2. Composite Band Construction (Average)
        b3 = (b1 + b2) / 2.0

        # 3. Stack to 3 Channels (H, W, C) for Albumentations
        img_processed = np.dstack((b1, b2, b3)).astype(np.float32)

        # 4. Augmentation (Resize, Flip, Rotate, ImageNet Norm)
        if self.transform:
            augmented = self.transform(image=img_processed)
            img_tensor = augmented["image"]
        else:
            # Fallback (should not happen)
            img_tensor = torch.from_numpy(img_processed.transpose(2, 0, 1))

        # 5. Normalize Incidence Angle
        # Using statistics from analysis: Mean ~39.28, Std ~3.84
        # We use a standard scaler approach
        angle_norm = (angle - 39.28) / 3.84

        sample = {
            "image": img_tensor,
            "angle": torch.tensor(angle_norm, dtype=torch.float32),
            "id": self.ids[idx] if self.ids is not None else "",
        }

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sample


def get_transforms(phase):
    """
    Returns the Albumentations transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                # Bicubic Upsampling
                A.Resize(
                    Config.IMG_HEIGHT, Config.IMG_WIDTH, interpolation=cv2.INTER_CUBIC
                ),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=Config.AUG_ROTATION_RANGE, p=0.5),  # Continuous rotation
                A.RandomRotate90(p=0.5),  # Discrete rotation
                # Normalization (ImageNet)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:  # val or test
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_HEIGHT, Config.IMG_WIDTH, interpolation=cv2.INTER_CUBIC
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_loaders(load_cached_data=True):
    """
    Loads metadata, processes raw data, and returns DataLoaders for Train, Val, and Test.
    """
    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw Data (Cached or Processed)
    train_data_full = load_and_process_json(
        Config.TRAIN_JSON, "train", load_cached_data
    )
    test_data_full = load_and_process_json(Config.TEST_JSON, "test", load_cached_data)

    # Helper to slice full data based on metadata indices
    def create_subset(meta_df, full_data, is_labeled=True):
        # Metadata contains 'sample_index' which maps directly to the raw list index
        indices = meta_df["sample_index"].values

        images = full_data["images"][indices]
        angles = full_data["angles"][indices]
        ids = meta_df["id"].values

        labels = None
        if is_labeled:
            labels = full_data["labels"][indices]

        return images, angles, labels, ids

    # Create Arrays
    X_train, ang_train, y_train, id_train = create_subset(
        train_meta, train_data_full, is_labeled=True
    )
    X_val, ang_val, y_val, id_val = create_subset(
        val_meta, train_data_full, is_labeled=True
    )
    X_test, ang_test, _, id_test = create_subset(
        test_meta, test_data_full, is_labeled=False
    )

    # Debug Slicing
    if Config.DEBUG:
        print("DEBUG MODE: Using small subset of data.")
        X_train, ang_train, y_train, id_train = (
            X_train[:128],
            ang_train[:128],
            y_train[:128],
            id_train[:128],
        )
        X_val, ang_val, y_val, id_val = (
            X_val[:64],
            ang_val[:64],
            y_val[:64],
            id_val[:64],
        )
        X_test, ang_test, id_test = X_test[:64], ang_test[:64], id_test[:64]

    # Create Datasets
    train_ds = IcebergDataset(
        X_train, ang_train, y_train, id_train, transform=get_transforms("train")
    )
    val_ds = IcebergDataset(
        X_val, ang_val, y_val, id_val, transform=get_transforms("val")
    )
    test_ds = IcebergDataset(
        X_test, ang_test, None, id_test, transform=get_transforms("test")
    )

    # Create DataLoaders
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
