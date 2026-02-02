import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library import config, utils

# ==========================================
# DATA PROCESSING & CACHING
# ==========================================


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, computes global statistics,
    and caches the result.

    Returns:
        train_data (dict): {'images': np.array, 'angles': np.array, 'labels': np.array, 'ids': np.array}
        test_data (dict): {'images': np.array, 'angles': np.array, 'ids': np.array}
        stats (dict): {'min': np.array, 'max': np.array, 'angle_mean': float}
    """

    # Check cache
    if load_cached_data and os.path.exists(config.CACHE_PATH):
        try:
            print(f"Loading cached data from {config.CACHE_PATH}...")
            cached = np.load(config.CACHE_PATH, allow_pickle=True)

            train_data = {
                "images": cached["train_images"],
                "angles": cached["train_angles"],
                "labels": cached["train_labels"],
                "ids": cached["train_ids"],
            }
            test_data = {
                "images": cached["test_images"],
                "angles": cached["test_angles"],
                "ids": cached["test_ids"],
            }
            stats = {
                "min": cached["stats_min"],
                "max": cached["stats_max"],
                "angle_mean": float(cached["stats_angle_mean"]),
            }
            return train_data, test_data, stats
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # Load Raw Data
    with open(config.TRAIN_JSON, "r") as f:
        train_json = json.load(f)
    with open(config.TEST_JSON, "r") as f:
        test_json = json.load(f)

    # Helper to process a list of dicts into arrays
    def process_json_list(data_list, is_train=True):
        ids = []
        band_1_list = []
        band_2_list = []
        angles = []
        labels = []

        for item in data_list:
            ids.append(item["id"])
            # Reshape flattened 5625 to 75x75
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            band_1_list.append(b1)
            band_2_list.append(b2)

            # Angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            if is_train:
                labels.append(item["is_iceberg"])

        # Stack bands: (N, 75, 75)
        b1_stack = np.stack(band_1_list)
        b2_stack = np.stack(band_2_list)
        # Compute Band 3 (Mean)
        b3_stack = (b1_stack + b2_stack) / 2.0

        # Stack channels: (N, 75, 75, 3)
        images = np.stack([b1_stack, b2_stack, b3_stack], axis=-1)

        ids = np.array(ids)
        angles = np.array(angles)

        if is_train:
            labels = np.array(labels)
            return images, angles, labels, ids
        else:
            return images, angles, ids

    # Process Train
    train_images, train_angles, train_labels, train_ids = process_json_list(
        train_json, is_train=True
    )

    # Process Test
    test_images, test_angles, test_ids = process_json_list(test_json, is_train=False)

    # Handle Incidence Angle Missing Values
    # Calculate mean from training set (ignoring NaNs)
    angle_mean = np.nanmean(train_angles)

    # Fill NaNs
    train_angles = np.nan_to_num(train_angles, nan=angle_mean)
    test_angles = np.nan_to_num(test_angles, nan=angle_mean)

    # Compute Global Normalization Statistics (Per Channel)
    # Shape of images: (N, 75, 75, 3)
    # We want min/max per channel across all pixels and all images in training set
    # Reshape to (N * 75 * 75, 3)
    flat_train = train_images.reshape(-1, 3)

    global_min = flat_train.min(axis=0)  # Shape (3,)
    global_max = flat_train.max(axis=0)  # Shape (3,)

    # Save to Cache
    os.makedirs(os.path.dirname(config.CACHE_PATH), exist_ok=True)
    np.savez(
        config.CACHE_PATH,
        train_images=train_images,
        train_angles=train_angles,
        train_labels=train_labels,
        train_ids=train_ids,
        test_images=test_images,
        test_angles=test_angles,
        test_ids=test_ids,
        stats_min=global_min,
        stats_max=global_max,
        stats_angle_mean=angle_mean,
    )

    train_data = {
        "images": train_images,
        "angles": train_angles,
        "labels": train_labels,
        "ids": train_ids,
    }
    test_data = {"images": test_images, "angles": test_angles, "ids": test_ids}
    stats = {"min": global_min, "max": global_max, "angle_mean": angle_mean}

    return train_data, test_data, stats


# ==========================================
# DATASET CLASS
# ==========================================


class IcebergDataset(Dataset):
    def __init__(self, images, angles, stats, labels=None, ids=None, transform=False):
        """
        Args:
            images (np.array): (N, 75, 75, 3)
            angles (np.array): (N,)
            stats (dict): {'min': np.array, 'max': np.array}
            labels (np.array, optional): (N,)
            ids (np.array, optional): (N,)
            transform (bool): Whether to apply augmentation
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.stats_min = stats["min"].reshape(1, 1, 3)
        self.stats_max = stats["max"].reshape(1, 1, 3)
        self.denom = self.stats_max - self.stats_min
        # Avoid division by zero just in case
        self.denom[self.denom == 0] = 1.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx].copy()  # (75, 75, 3)
        angle = self.angles[idx]

        # Apply Global Min-Max Scaling
        # (img - min) / (max - min)
        # Note: We do NOT clip, allowing outliers in validation/test.
        img = (img - self.stats_min) / self.denom

        # Augmentation (Train only)
        if self.transform:
            # Random Rotation: 0, 90, 180, 270
            k = np.random.randint(0, 4)
            img = np.rot90(img, k=k, axes=(0, 1))

            # Random Horizontal Flip
            if np.random.random() < 0.5:
                img = np.fliplr(img)

        # Convert to Tensor
        # Numpy is (H, W, C) -> PyTorch (C, H, W)
        img_tensor = torch.from_numpy(img.copy()).float().permute(2, 0, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Return
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            # For training/validation, we typically just need inputs and targets
            if self.ids is not None:
                return img_tensor, angle_tensor, label_tensor, self.ids[idx]
            return img_tensor, angle_tensor, label_tensor
        else:
            # For testing, we need IDs for submission
            if self.ids is not None:
                return img_tensor, angle_tensor, self.ids[idx]
            return img_tensor, angle_tensor


# ==========================================
# DATA LOADING
# ==========================================


def get_loaders(fold_idx, load_cached_data=True, debug_size=None):
    """
    Creates DataLoaders for a specific fold using Stratified K-Fold.

    Args:
        fold_idx (int): Index of the fold (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached preprocessed data.
        debug_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Get Data
    train_data, test_data, stats = process_and_cache_data(load_cached_data)

    images = train_data["images"]
    angles = train_data["angles"]
    labels = train_data["labels"]
    ids = train_data["ids"]

    # Debugging: Truncate data if needed
    if debug_size is not None:
        images = images[:debug_size]
        angles = angles[:debug_size]
        labels = labels[:debug_size]
        ids = ids[:debug_size]

    # 2. Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Get indices for the requested fold
    fold_generator = skf.split(images, labels)
    train_indices, val_indices = next(
        x for i, x in enumerate(fold_generator) if i == fold_idx
    )

    # 3. Create Datasets
    # Train Dataset: Augmentation = True, Ids = None (standard format)
    train_dataset = IcebergDataset(
        images=images[train_indices],
        angles=angles[train_indices],
        labels=labels[train_indices],
        ids=ids[train_indices],
        stats=stats,
        transform=True,
    )

    # Validation Dataset: Augmentation = False, Ids = None
    val_dataset = IcebergDataset(
        images=images[val_indices],
        angles=angles[val_indices],
        labels=labels[val_indices],
        ids=ids[val_indices],
        stats=stats,
        transform=False,
    )

    # Test Dataset: Augmentation = False, Ids = Provided (for submission)
    test_dataset = IcebergDataset(
        images=test_data["images"],
        angles=test_data["angles"],
        labels=None,
        ids=test_data["ids"],
        stats=stats,
        transform=False,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

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
