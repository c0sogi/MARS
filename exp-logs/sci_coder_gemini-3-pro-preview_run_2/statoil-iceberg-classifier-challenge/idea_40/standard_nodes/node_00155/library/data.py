import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    CACHE_DIR,
    IMAGE_SIZE,
    NUM_FOLDS,
    BATCH_SIZE,
    SEED,
    NUM_WORKERS,
    USE_GLOBAL_SCALING,
)
from library.utils import set_seed


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw data, processes it (parsing, cleaning, stats), and caches it.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary containing processed numpy arrays and statistics.
    """
    cache_npz_path = os.path.join(CACHE_DIR, "processed_data.npz")
    cache_stats_path = os.path.join(CACHE_DIR, "stats.json")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_npz_path)
        and os.path.exists(cache_stats_path)
    ):
        try:
            print(f"Loading cached data from {CACHE_DIR}...")
            npz_data = np.load(cache_npz_path)
            with open(cache_stats_path, "r") as f:
                stats = json.load(f)

            return {
                "train_images": npz_data["train_images"],
                "train_angles": npz_data["train_angles"],
                "train_labels": npz_data["train_labels"],
                "train_ids": npz_data["train_ids"],
                "test_images": npz_data["test_images"],
                "test_angles": npz_data["test_angles"],
                "test_ids": npz_data["test_ids"],
                "stats": stats,
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch...")
    else:
        print("Processing data from scratch...")

    # Load raw JSON files
    train_path = os.path.join(INPUT_DIR, "train.json")
    test_path = os.path.join(INPUT_DIR, "test.json")

    with open(train_path, "r") as f:
        train_data_raw = json.load(f)
    with open(test_path, "r") as f:
        test_data_raw = json.load(f)

    # Helper function to process raw list of dicts
    def process_raw_list(data_list, has_labels=True):
        ids = []
        band_1_list = []
        band_2_list = []
        angles = []
        labels = []

        for item in data_list:
            ids.append(item["id"])

            # Reshape flattened 5625 list to 75x75
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
            band_1_list.append(b1)
            band_2_list.append(b2)

            # Handle incidence angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            if has_labels:
                labels.append(item["is_iceberg"])

        # Stack bands to (N, 75, 75, 2)
        images = np.stack([np.array(band_1_list), np.array(band_2_list)], axis=-1)
        angles = np.array(angles, dtype=np.float32)
        ids = np.array(ids).flatten()

        if has_labels:
            labels = np.array(labels, dtype=np.float32)
            return images, angles, labels, ids
        else:
            return images, angles, ids

    # Process datasets
    train_images, train_angles, train_labels, train_ids = process_raw_list(
        train_data_raw, has_labels=True
    )
    test_images, test_angles, test_ids = process_raw_list(
        test_data_raw, has_labels=False
    )

    # Impute missing incidence angles using mean of valid training angles
    angle_mean = np.nanmean(train_angles)
    train_angles_filled = np.where(np.isnan(train_angles), angle_mean, train_angles)
    test_angles_filled = np.where(np.isnan(test_angles), angle_mean, test_angles)

    # Compute Global Statistics for Scaling (from Training Set only)
    # We need min/max for Band 1, Band 2, and the derived Mean Band (B1+B2)/2
    b1_train = train_images[..., 0]
    b2_train = train_images[..., 1]
    b3_train = (b1_train + b2_train) / 2.0

    stats = {
        "min_b1": float(np.min(b1_train)),
        "max_b1": float(np.max(b1_train)),
        "min_b2": float(np.min(b2_train)),
        "max_b2": float(np.max(b2_train)),
        "min_b3": float(np.min(b3_train)),
        "max_b3": float(np.max(b3_train)),
        "angle_mean": float(angle_mean),
    }

    # Save to cache
    np.savez(
        cache_npz_path,
        train_images=train_images,
        train_angles=train_angles_filled,
        train_labels=train_labels,
        train_ids=train_ids,
        test_images=test_images,
        test_angles=test_angles_filled,
        test_ids=test_ids,
    )

    with open(cache_stats_path, "w") as f:
        json.dump(stats, f)

    print(f"Data processed and cached at {CACHE_DIR}")

    return {
        "train_images": train_images,
        "train_angles": train_angles_filled,
        "train_labels": train_labels,
        "train_ids": train_ids,
        "test_images": test_images,
        "test_angles": test_angles_filled,
        "test_ids": test_ids,
        "stats": stats,
    }


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=False, stats=None):
        """
        Args:
            images: np.ndarray (N, 75, 75, 2)
            angles: np.ndarray (N,)
            labels: np.ndarray (N,) or None
            transform: bool, whether to apply augmentations
            stats: dict, global scaling statistics
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform
        self.stats = stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Extract data
        img = self.images[idx]  # (75, 75, 2)
        angle = self.angles[idx]

        # Separate bands
        b1 = img[..., 0]
        b2 = img[..., 1]

        # Construct 3rd channel: Arithmetic Mean
        b3 = (b1 + b2) / 2.0

        # Apply Global Min-Max Scaling
        if self.stats:
            b1 = (b1 - self.stats["min_b1"]) / (
                self.stats["max_b1"] - self.stats["min_b1"]
            )
            b2 = (b2 - self.stats["min_b2"]) / (
                self.stats["max_b2"] - self.stats["min_b2"]
            )
            b3 = (b3 - self.stats["min_b3"]) / (
                self.stats["max_b3"] - self.stats["min_b3"]
            )

        # Stack to create 3-channel image (75, 75, 3)
        img_processed = np.dstack((b1, b2, b3))

        # Apply Augmentations (Train only)
        if self.transform:
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                img_processed = np.fliplr(img_processed)

            # Random Rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            img_processed = np.rot90(img_processed, k=k)

        # Convert to Tensor (C, H, W)
        # .copy() is required because rot90/flip can return negative strides which torch doesn't support
        img_tensor = torch.from_numpy(img_processed.copy()).float().permute(2, 0, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # Return -1 for test set labels
            return img_tensor, angle_tensor, torch.tensor(-1.0)


def get_data_loaders(fold_index=0, load_cached_data=True, debug=False):
    """
    Creates DataLoaders for a specific fold of Stratified K-Fold CV.

    Args:
        fold_index (int): The fold to use for validation (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Ensure reproducibility
    set_seed(SEED)

    # Get processed data
    data = process_and_cache_data(load_cached_data=load_cached_data)

    train_images = data["train_images"]
    train_angles = data["train_angles"]
    train_labels = data["train_labels"]

    test_images = data["test_images"]
    test_angles = data["test_angles"]
    test_ids = data["test_ids"]
    stats = data["stats"]

    # Debug Mode: Slice data to small subset
    if debug:
        print("DEBUG MODE: Truncating datasets to 100 samples.")
        subset_size = 100
        train_images = train_images[:subset_size]
        train_angles = train_angles[:subset_size]
        train_labels = train_labels[:subset_size]
        test_images = test_images[:subset_size]
        test_angles = test_angles[:subset_size]
        test_ids = test_ids[:subset_size]

    # Stratified K-Fold Split
    # We split based on the available training data
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Generate splits and select the requested fold
    splits = list(skf.split(train_images, train_labels))
    if fold_index < 0 or fold_index >= len(splits):
        raise ValueError(
            f"Fold index {fold_index} must be between 0 and {len(splits)-1}"
        )

    train_idx, val_idx = splits[fold_index]

    # Create Train/Val subsets
    X_train, X_val = train_images[train_idx], train_images[val_idx]
    a_train, a_val = train_angles[train_idx], train_angles[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]

    # Instantiate Datasets
    # Train: Augmentation ON
    train_dataset = IcebergDataset(
        X_train, a_train, y_train, transform=True, stats=stats
    )

    # Val: Augmentation OFF
    val_dataset = IcebergDataset(X_val, a_val, y_val, transform=False, stats=stats)

    # Test: Augmentation OFF
    test_dataset = IcebergDataset(
        test_images, test_angles, labels=None, transform=False, stats=stats
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

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

    return train_loader, val_loader, test_loader, test_ids
