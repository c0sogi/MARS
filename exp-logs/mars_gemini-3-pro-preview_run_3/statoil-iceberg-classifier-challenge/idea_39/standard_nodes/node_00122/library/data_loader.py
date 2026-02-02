import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    IMAGE_SIZE,
)


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        # Input X is (3, 75, 75) float32
        img = torch.from_numpy(self.X[idx])

        # Apply augmentations if provided
        if self.transform:
            img = self.transform(img)

        # Process angle
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Process label
        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
        else:
            # Dummy label for test set
            label = torch.tensor([0.0], dtype=torch.float32)

        # Get ID
        sample_id = self.ids[idx]

        return img, angle, label, sample_id


def process_bands(data_list):
    """
    Extracts band_1 and band_2, reshapes them, calculates band_3 (avg),
    and stacks them into a (N, 3, 75, 75) array.
    """
    count = len(data_list)
    # Pre-allocate array for efficiency
    # Shape: (N, Channels, Height, Width)
    X = np.empty((count, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for i, item in enumerate(data_list):
        # Band 1 (HH)
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
        # Band 2 (HV)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
        # Band 3 (Average)
        b3 = (b1 + b2) / 2.0

        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

    return X


def load_data_split(meta_path, json_path, cache_prefix, load_cached_data=True):
    """
    Loads data for a specific split (train, val, or test).
    Uses caching to avoid re-processing JSON files.
    """
    # Define cache file paths
    cache_X = os.path.join(CACHE_DIR, f"X_{cache_prefix}.npy")
    cache_ang = os.path.join(CACHE_DIR, f"angle_{cache_prefix}.npy")
    cache_y = os.path.join(CACHE_DIR, f"y_{cache_prefix}.npy")
    cache_id = os.path.join(CACHE_DIR, f"ids_{cache_prefix}.npy")

    # Check if cache exists
    files_exist = (
        os.path.exists(cache_X)
        and os.path.exists(cache_ang)
        and os.path.exists(cache_id)
        and (os.path.exists(cache_y) or cache_prefix == "test")
    )

    if load_cached_data and files_exist:
        print(f"Loading {cache_prefix} data from cache...")
        X = np.load(cache_X)
        angles = np.load(cache_ang)
        ids = np.load(cache_id)
        if cache_prefix != "test":
            y = np.load(cache_y)
        else:
            y = None
        return X, angles, y, ids

    print(f"Processing {cache_prefix} data from raw files...")

    # Load Metadata
    df_meta = pd.read_csv(meta_path)

    # Load Raw JSON
    # Note: This loads the entire JSON. For very large datasets, we might need chunking,
    # but for this task (1600 train, 300 test), it fits in memory easily.
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Map raw data by original index for fast retrieval
    # The metadata contains 'original_index' which corresponds to the index in the raw json list
    indices = df_meta["original_index"].values
    subset_data = [raw_data[i] for i in indices]

    # Process Images
    X = process_bands(subset_data)

    # Process Angles (Use metadata as it already handles 'na' -> NaN conversion)
    angles = df_meta["inc_angle"].values.astype(np.float32)

    # Process IDs
    ids = df_meta["id"].values

    # Process Labels
    if "is_iceberg" in df_meta.columns:
        y = df_meta["is_iceberg"].values.astype(np.float32)
    else:
        y = None

    # Save to cache
    np.save(cache_X, X)
    np.save(cache_ang, angles)
    np.save(cache_id, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, angles, y, ids


def seed_worker(worker_id):
    """
    Worker initialization function to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


def get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Performs imputation on incidence angles using training set median.
    """

    # 1. Load Data Splits
    X_train, ang_train, y_train, id_train = load_data_split(
        TRAIN_META_PATH, TRAIN_JSON, "train", load_cached_data
    )
    X_val, ang_val, y_val, id_val = load_data_split(
        VAL_META_PATH, TRAIN_JSON, "val", load_cached_data
    )
    X_test, ang_test, _, id_test = load_data_split(
        TEST_META_PATH, TEST_JSON, "test", load_cached_data
    )

    # 2. Impute Missing Incidence Angles
    # Compute median from training set (ignoring NaNs)
    angle_median = np.nanmedian(ang_train)

    # Fill NaNs
    ang_train = np.where(np.isnan(ang_train), angle_median, ang_train)
    ang_val = np.where(np.isnan(ang_val), angle_median, ang_val)
    ang_test = np.where(np.isnan(ang_test), angle_median, ang_test)

    # 3. Define Transforms
    # Train: Horizontal and Vertical Flips
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Val/Test: No augmentation
    val_transform = None

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, id_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, id_val, transform=val_transform)
    test_dataset = IcebergDataset(
        X_test, ang_test, None, id_test, transform=val_transform
    )

    # 5. Create DataLoaders
    # Use a generator with fixed seed for reproducibility in shuffling
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
