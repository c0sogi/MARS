import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): String IDs of shape (N,).
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
        # Load data
        img = self.X[idx]  # (75, 75, 3)
        angle = self.angles[idx]

        # Convert to tensor
        # PyTorch expects (C, H, W), but numpy is (H, W, C)
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, 75, 75)

        # Apply augmentations
        if self.transform:
            img_tensor = self.transform(img_tensor)

        sample = {
            "image": img_tensor,
            "angle": torch.tensor(angle, dtype=torch.float32),
        }

        if self.y is not None:
            sample["label"] = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw data, processes it (reshape, 3rd channel, angle imputation),
    and caches it as numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing train, val, and test data arrays.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train.npy"),
        "ids_train": os.path.join(Config.CACHE_DIR, "ids_train.npy"),
        "X_val": os.path.join(Config.CACHE_DIR, "X_val.npy"),
        "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
        "angle_val": os.path.join(Config.CACHE_DIR, "angle_val.npy"),
        "ids_val": os.path.join(Config.CACHE_DIR, "ids_val.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if all files exist
    all_cached = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        data = {k: np.load(v, allow_pickle=True) for k, v in files.items()}
    else:
        print("Processing data from scratch...")

        # Load Metadata
        train_meta = pd.read_csv(Config.TRAIN_META)
        val_meta = pd.read_csv(Config.VAL_META)
        test_meta = pd.read_csv(Config.TEST_META)

        # Load Raw JSONs
        # Note: This is memory intensive, but fits within 220GB RAM
        print("Loading raw train.json...")
        with open(Config.TRAIN_JSON, "r") as f:
            raw_train_data = json.load(f)
            # Create a lookup dictionary for O(1) access by id
            # or simply list since metadata has original_index

        # We can use original_index from metadata to index into the list directly
        # assuming the list order hasn't changed. The metadata generation script
        # preserved original_index.

        print("Loading raw test.json...")
        with open(Config.TEST_JSON, "r") as f:
            raw_test_data = json.load(f)

        # Helper to process a subset
        def process_subset(meta_df, raw_source, has_target=True):
            indices = meta_df["original_index"].values
            ids = meta_df["id"].values

            # Pre-allocate arrays
            n_samples = len(indices)
            X = np.zeros((n_samples, 75, 75, 3), dtype=np.float32)
            angles = np.full(n_samples, np.nan, dtype=np.float32)
            y = np.zeros(n_samples, dtype=np.float32) if has_target else None

            for i, original_idx in enumerate(indices):
                item = raw_source[original_idx]

                # Verify ID match to be safe
                if item["id"] != ids[i]:
                    raise ValueError(
                        f"ID mismatch at index {i}: Meta {ids[i]} vs Raw {item['id']}"
                    )

                # Process Images
                b1 = np.array(item["band_1"]).reshape(75, 75)
                b2 = np.array(item["band_2"]).reshape(75, 75)
                avg = (b1 + b2) / 2.0

                # Stack channels: (75, 75, 3)
                X[i, :, :, 0] = b1
                X[i, :, :, 1] = b2
                X[i, :, :, 2] = avg

                # Process Angle
                # Metadata already has numeric conversion, but let's take from raw or meta
                # Using meta is safer as it handled 'na'
                ang = meta_df.iloc[i]["inc_angle"]
                angles[i] = ang

                if has_target:
                    y[i] = item["is_iceberg"]

            return X, angles, y, ids

        print("Processing Train split...")
        X_train, angle_train, y_train, ids_train = process_subset(
            train_meta, raw_train_data, True
        )

        print("Processing Val split...")
        X_val, angle_val, y_val, ids_val = process_subset(
            val_meta, raw_train_data, True
        )

        print("Processing Test split...")
        X_test, angle_test, _, ids_test = process_subset(
            test_meta, raw_test_data, False
        )

        # Impute Missing Angles
        # Calculate median from TRAIN set only
        angle_median = np.nanmedian(angle_train)
        print(f"Imputing missing angles with median: {angle_median:.4f}")

        # Apply imputation
        angle_train = np.where(np.isnan(angle_train), angle_median, angle_train)
        angle_val = np.where(np.isnan(angle_val), angle_median, angle_val)
        angle_test = np.where(np.isnan(angle_test), angle_median, angle_test)

        # Save to cache
        print("Saving to cache...")
        np.save(files["X_train"], X_train)
        np.save(files["y_train"], y_train)
        np.save(files["angle_train"], angle_train)
        np.save(files["ids_train"], ids_train)

        np.save(files["X_val"], X_val)
        np.save(files["y_val"], y_val)
        np.save(files["angle_val"], angle_val)
        np.save(files["ids_val"], ids_val)

        np.save(files["X_test"], X_test)
        np.save(files["angle_test"], angle_test)
        np.save(files["ids_test"], ids_test)

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "angle_train": angle_train,
            "ids_train": ids_train,
            "X_val": X_val,
            "y_val": y_val,
            "angle_val": angle_val,
            "ids_val": ids_val,
            "X_test": X_test,
            "angle_test": angle_test,
            "ids_test": ids_test,
        }

        # Clean up memory
        del raw_train_data, raw_test_data

    # Debug Mode Slicing
    if Config.DEBUG:
        limit = Config.MAX_DEBUG_SAMPLES
        print(f"DEBUG: Truncating datasets to {limit} samples.")
        for k in data:
            if len(data[k]) > limit:
                data[k] = data[k][:limit]

    return data


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get processed data
    data = process_and_cache_data(load_cached_data)

    # Define Transforms
    # Note: Input is (C, H, W) tensor.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # No TTA for val/test as per requirements
    val_transform = None
    test_transform = None

    # Create Datasets
    train_dataset = IcebergDataset(
        X=data["X_train"],
        angles=data["angle_train"],
        y=data["y_train"],
        ids=data["ids_train"],
        transform=train_transform,
    )

    val_dataset = IcebergDataset(
        X=data["X_val"],
        angles=data["angle_val"],
        y=data["y_val"],
        ids=data["ids_val"],
        transform=val_transform,
    )

    test_dataset = IcebergDataset(
        X=data["X_test"],
        angles=data["angle_test"],
        y=None,
        ids=data["ids_test"],
        transform=test_transform,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
