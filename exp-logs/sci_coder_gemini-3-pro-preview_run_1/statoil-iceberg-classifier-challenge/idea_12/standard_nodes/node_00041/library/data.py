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


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Base transforms: Resize (Bicubic) and Normalize (ImageNet)
    transforms_list = [
        A.Resize(
            height=Config.IMG_SIZE, width=Config.IMG_SIZE, interpolation=cv2.INTER_CUBIC
        )
    ]

    if phase == "train":
        # Geometric Augmentations for Training
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=Config.ROTATION_LIMIT, p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

    # Normalization and Tensor conversion
    transforms_list.extend(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,  # Input is already [0, 1]
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays (3-channel images),
    and caches the results. Implements the 'Normalize then Average' fusion strategy.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        dict: Contains 'train_images', 'train_angles', 'train_labels',
              'test_images', 'test_angles', 'angle_stats'.
    """
    # Define cache paths
    cache_files = {
        "train_images": os.path.join(Config.WORKING_DIR, "train_images.npy"),
        "train_angles": os.path.join(Config.WORKING_DIR, "train_angles.npy"),
        "train_labels": os.path.join(Config.WORKING_DIR, "train_labels.npy"),
        "test_images": os.path.join(Config.WORKING_DIR, "test_images.npy"),
        "test_angles": os.path.join(Config.WORKING_DIR, "test_angles.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
        "stats": os.path.join(Config.WORKING_DIR, "stats.json"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            data = {}
            for k, v in cache_files.items():
                if k == "stats":
                    with open(v, "r") as f:
                        data[k] = json.load(f)
                else:
                    data[k] = np.load(v, allow_pickle=True)  # allow_pickle for IDs
            return data

    print("Cache missing or reload requested. Processing raw data...")

    # 2. Load Raw Data
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        raw_test = json.load(f)

    # 3. Process Training Data
    # Extract Bands
    train_b1 = np.array([x["band_1"] for x in raw_train], dtype=np.float32).reshape(
        -1, 75, 75
    )
    train_b2 = np.array([x["band_2"] for x in raw_train], dtype=np.float32).reshape(
        -1, 75, 75
    )
    train_labels = np.array([x["is_iceberg"] for x in raw_train], dtype=np.float32)

    # Extract Angles (Handle 'na')
    train_angles = []
    valid_angles = []
    for x in raw_train:
        a = x["inc_angle"]
        if a == "na":
            train_angles.append(np.nan)
        else:
            val = float(a)
            train_angles.append(val)
            valid_angles.append(val)

    train_angles = np.array(train_angles, dtype=np.float32)

    # Compute Angle Stats (Mean/Std) for imputation and normalization
    angle_mean = np.mean(valid_angles)
    angle_std = np.std(valid_angles)

    # Impute 'na' in train_angles
    train_angles[np.isnan(train_angles)] = angle_mean

    # Compute Image Stats (Global Min/Max) from Training Data
    b1_min, b1_max = train_b1.min(), train_b1.max()
    b2_min, b2_max = train_b2.min(), train_b2.max()

    stats = {
        "angle_mean": float(angle_mean),
        "angle_std": float(angle_std),
        "b1_min": float(b1_min),
        "b1_max": float(b1_max),
        "b2_min": float(b2_min),
        "b2_max": float(b2_max),
    }

    # Helper for Image Normalization and Stacking
    def create_3ch_image(b1, b2, stats):
        # Normalize to [0, 1]
        b1_norm = (b1 - stats["b1_min"]) / (stats["b1_max"] - stats["b1_min"])
        b2_norm = (b2 - stats["b2_min"]) / (stats["b2_max"] - stats["b2_min"])

        # Composite Band (Average)
        b3_avg = (b1_norm + b2_norm) / 2.0

        # Stack: (N, 75, 75, 3)
        return np.stack([b1_norm, b2_norm, b3_avg], axis=-1)

    train_images = create_3ch_image(train_b1, train_b2, stats)

    # 4. Process Test Data
    test_b1 = np.array([x["band_1"] for x in raw_test], dtype=np.float32).reshape(
        -1, 75, 75
    )
    test_b2 = np.array([x["band_2"] for x in raw_test], dtype=np.float32).reshape(
        -1, 75, 75
    )
    test_ids = np.array([x["id"] for x in raw_test])

    # Test Angles (Assume no 'na' or handle if present, though description says no 'na')
    test_angles = []
    for x in raw_test:
        try:
            test_angles.append(float(x["inc_angle"]))
        except (ValueError, TypeError):
            test_angles.append(angle_mean)  # Fallback
    test_angles = np.array(test_angles, dtype=np.float32)

    test_images = create_3ch_image(test_b1, test_b2, stats)

    # 5. Save to Cache
    np.save(cache_files["train_images"], train_images)
    np.save(cache_files["train_angles"], train_angles)
    np.save(cache_files["train_labels"], train_labels)
    np.save(cache_files["test_images"], test_images)
    np.save(cache_files["test_angles"], test_angles)
    np.save(cache_files["test_ids"], test_ids)

    with open(cache_files["stats"], "w") as f:
        json.dump(stats, f)

    print("Data processed and cached.")

    return {
        "train_images": train_images,
        "train_angles": train_angles,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_angles": test_angles,
        "test_ids": test_ids,
        "stats": stats,
    }


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles on-the-fly augmentation and angle normalization.
    """

    def __init__(
        self, images, angles, labels=None, ids=None, transform=None, angle_stats=None
    ):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.angle_stats = angle_stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get Image (H, W, C)
        image = self.images[idx]

        # 2. Apply Augmentations (Albumentations)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to ToTensor if no transform provided (should not happen in pipeline)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # 3. Get and Normalize Angle
        angle = self.angles[idx]
        if self.angle_stats:
            angle = (angle - self.angle_stats["angle_mean"]) / self.angle_stats[
                "angle_std"
            ]

        angle = torch.tensor(angle, dtype=torch.float32)

        # 4. Return tuple
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            # Test mode
            id_val = self.ids[idx]
            return image, angle, id_val


def get_dataloaders(load_cached_data=True, full_fit=False):
    """
    Constructs DataLoaders for Train, Val (optional), and Test.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        full_fit (bool): If True, merges Train and Val into a single training loader (Phase 2).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
               val_loader will be None if full_fit is True.
    """
    # 1. Load Data Arrays
    data = process_and_cache_data(load_cached_data=load_cached_data)

    # 2. Load Metadata (indices)
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 3. Prepare Indices
    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values
    test_indices = df_test_meta["sample_index"].values

    # 4. Construct Datasets

    # Training Set
    if full_fit:
        # Combine indices for Phase 2
        combined_indices = np.concatenate([train_indices, val_indices])
        train_dataset = IcebergDataset(
            images=data["train_images"][combined_indices],
            angles=data["train_angles"][combined_indices],
            labels=data["train_labels"][combined_indices],
            transform=get_transforms("train"),
            angle_stats=data["stats"],
        )
        val_loader = None
    else:
        # Standard Split for Phase 1
        train_dataset = IcebergDataset(
            images=data["train_images"][train_indices],
            angles=data["train_angles"][train_indices],
            labels=data["train_labels"][train_indices],
            transform=get_transforms("train"),
            angle_stats=data["stats"],
        )

        val_dataset = IcebergDataset(
            images=data["train_images"][val_indices],
            angles=data["train_angles"][val_indices],
            labels=data["train_labels"][val_indices],
            transform=get_transforms("val"),
            angle_stats=data["stats"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    # Test Set
    # Note: test_indices in metadata map to the order in test.json, which matches data['test_images'] order.
    # We use indices to be safe, though test.json is usually read sequentially.
    test_dataset = IcebergDataset(
        images=data["test_images"][test_indices],
        angles=data["test_angles"][test_indices],
        ids=data["test_ids"][test_indices],
        transform=get_transforms("test"),
        angle_stats=data["stats"],
    )

    # 5. Construct Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
