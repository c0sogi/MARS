import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def process_and_cache_data(load_cached_data=True):
    """
    Handles data loading, preprocessing (3-channel construction, normalization),
    and caching.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: Contains training, validation, and test arrays.
    """
    cache_path = Config.CACHE_PATH

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw JSON Data
    # We load both files into a dictionary for O(1) lookup by ID
    print("Loading raw JSON files...")
    with open(Config.TRAIN_JSON, "r") as f:
        train_json_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_json_data = json.load(f)

    # Map ID to data dict
    id_to_data = {item["id"]: item for item in train_json_data}
    id_to_data.update({item["id"]: item for item in test_json_data})

    def extract_features(df_meta, is_test=False):
        ids = df_meta["id"].values
        count = len(ids)

        # Pre-allocate arrays
        # Shape: (N, 3, 75, 75) for PyTorch (C, H, W)
        images = np.zeros((count, 3, 75, 75), dtype=np.float32)
        angles = np.zeros(count, dtype=np.float32)
        labels = np.zeros(count, dtype=np.float32) if not is_test else None

        for i, img_id in enumerate(ids):
            item = id_to_data[img_id]

            # Extract Bands
            band_1 = np.array(item["band_1"]).reshape(75, 75)
            band_2 = np.array(item["band_2"]).reshape(75, 75)

            # Channel 1: HH (Band 1)
            images[i, 0, :, :] = band_1
            # Channel 2: HV (Band 2)
            images[i, 1, :, :] = band_2
            # Channel 3: Avg
            images[i, 2, :, :] = (band_1 + band_2) / 2.0

            # Extract Angle
            # Handle 'na' by converting to float (becomes NaN)
            try:
                angle = float(item["inc_angle"])
            except (ValueError, TypeError):
                angle = np.nan
            angles[i] = angle

            # Extract Label
            if not is_test:
                labels[i] = item["is_iceberg"]

        return ids, images, angles, labels

    print("Constructing arrays...")
    ids_train, X_train, ang_train, y_train = extract_features(
        df_train_meta, is_test=False
    )
    ids_val, X_val, ang_val, y_val = extract_features(df_val_meta, is_test=False)
    ids_test, X_test, ang_test, _ = extract_features(df_test_meta, is_test=True)

    # 3. Handle Missing Incidence Angles
    # Impute with mean of training set
    train_angle_mean = np.nanmean(ang_train)

    # Fill NaNs
    ang_train = np.nan_to_num(ang_train, nan=train_angle_mean)
    ang_val = np.nan_to_num(ang_val, nan=train_angle_mean)
    ang_test = np.nan_to_num(ang_test, nan=train_angle_mean)

    # 4. Normalization: Independent Per-Channel Min-Max Scaling
    # Calculate stats on Training set ONLY
    print("Normalizing data...")
    for c in range(3):
        # Extract channel c from training data
        train_c = X_train[:, c, :, :]

        min_val = np.min(train_c)
        max_val = np.max(train_c)

        # Avoid division by zero
        denom = max_val - min_val
        if denom == 0:
            denom = 1.0

        # Apply to Train
        X_train[:, c, :, :] = (X_train[:, c, :, :] - min_val) / denom
        # Apply to Val
        X_val[:, c, :, :] = (X_val[:, c, :, :] - min_val) / denom
        # Apply to Test
        X_test[:, c, :, :] = (X_test[:, c, :, :] - min_val) / denom

    # 5. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=X_train,
        ang_train=ang_train,
        y_train=y_train,
        ids_train=ids_train,
        X_val=X_val,
        ang_val=ang_val,
        y_val=y_val,
        ids_val=ids_val,
        X_test=X_test,
        ang_test=ang_test,
        ids_test=ids_test,
    )

    return {
        "X_train": X_train,
        "ang_train": ang_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_val": X_val,
        "ang_val": ang_val,
        "y_val": y_val,
        "ids_val": ids_val,
        "X_test": X_test,
        "ang_test": ang_test,
        "ids_test": ids_test,
    }


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            ids (np.ndarray, optional): Shape (N,)
            transform (bool): Whether to apply augmentations.
        """
        self.images = torch.from_numpy(images).float()
        self.angles = torch.from_numpy(angles).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        angle = self.angles[idx]

        # Apply Augmentation (Native Tensor Operations)
        if self.transform:
            # Random Horizontal Flip
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])  # [C, H, W], flip W

            # Random Rotation (0, 90, 180, 270)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])  # [C, H, W], rotate H, W

        if self.labels is not None:
            return img, angle, self.labels[idx]
        else:
            return img, angle, self.ids[idx]


def get_loaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data = process_and_cache_data(load_cached_data=load_cached_data)

    # Create Datasets
    # Train: Augmentation Enabled
    train_dataset = IcebergDataset(
        data["X_train"],
        data["ang_train"],
        data["y_train"],
        data["ids_train"],
        transform=True,
    )

    # Val: No Augmentation
    val_dataset = IcebergDataset(
        data["X_val"], data["ang_val"], data["y_val"], data["ids_val"], transform=False
    )

    # Test: No Augmentation
    test_dataset = IcebergDataset(
        data["X_test"],
        data["ang_test"],
        labels=None,
        ids=data["ids_test"],
        transform=False,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
