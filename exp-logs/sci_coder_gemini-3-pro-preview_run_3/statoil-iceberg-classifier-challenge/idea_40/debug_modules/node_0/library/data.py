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

    def __init__(self, X, angles, labels_or_ids, transform=None, is_test=False):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            labels_or_ids (np.ndarray): Labels (0/1) for train/val, or IDs for test.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is the test set (returns ID instead of label).
        """
        self.X = X
        self.angles = angles
        self.labels_or_ids = labels_or_ids
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]
        target = self.labels_or_ids[idx]

        # Convert to Tensor
        # Input data is float (dB), not uint8, so we convert directly to float tensor.
        img_tensor = torch.from_numpy(img).float()

        # Apply transforms (Augmentation)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Convert angle to tensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.is_test:
            # Return ID as is (string or object)
            return img_tensor, angle_tensor, target
        else:
            # Return label as float tensor for BCEWithLogitsLoss
            label_tensor = torch.tensor(target, dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor


def _load_and_process(
    metadata_path, raw_json_path, prefix, angle_impute_val, load_cached_data=True
):
    """
    Internal helper to load, process, and cache data.
    """
    # Define cache paths
    cache_X = os.path.join(Config.CACHE_DIR, f"{prefix}_X.npy")
    cache_angle = os.path.join(Config.CACHE_DIR, f"{prefix}_angle.npy")
    cache_y = os.path.join(Config.CACHE_DIR, f"{prefix}_y.npy")  # Stores labels or IDs

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_X)
        and os.path.exists(cache_angle)
        and os.path.exists(cache_y)
    ):
        print(f"Loading cached {prefix} data from {Config.CACHE_DIR}...")
        X = np.load(cache_X)
        angles = np.load(cache_angle)
        y = np.load(
            cache_y, allow_pickle=True
        )  # allow_pickle=True for string IDs in test set
        return X, angles, y

    # 2. Process from scratch
    print(f"Processing {prefix} data from scratch...")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)

    # Load raw JSON
    # Note: Loading large JSONs can be memory intensive, but fits in 220GB RAM.
    with open(raw_json_path, "r") as f:
        raw_data_list = json.load(f)

    # Create a lookup dictionary for O(1) access by ID
    id_to_data = {item["id"]: item for item in raw_data_list}

    X_list = []
    angles_list = []
    y_list = []

    is_test = "is_iceberg" not in df_meta.columns

    for _, row in df_meta.iterrows():
        img_id = row["id"]
        data_item = id_to_data[img_id]

        # --- Image Processing ---
        # Band 1 (HH) and Band 2 (HV) are flattened 5625 lists -> reshape to 75x75
        b1 = np.array(data_item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(data_item["band_2"], dtype=np.float32).reshape(75, 75)

        # Band 3 (Average)
        b3 = (b1 + b2) / 2.0

        # Stack channels: (3, 75, 75)
        img_stacked = np.stack([b1, b2, b3], axis=0)
        X_list.append(img_stacked)

        # --- Angle Processing ---
        angle = row["inc_angle"]
        if pd.isna(angle) or angle == "na":
            angle = angle_impute_val
        angles_list.append(float(angle))

        # --- Label/ID Processing ---
        if is_test:
            y_list.append(img_id)
        else:
            y_list.append(row["is_iceberg"])

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    angles = np.array(angles_list, dtype=np.float32)

    if is_test:
        y = np.array(y_list)  # Array of strings
    else:
        y = np.array(y_list, dtype=np.float32)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_X, X)
    np.save(cache_angle, angles)
    np.save(cache_y, y)

    return X, angles, y


def get_loaders(
    batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=True
):
    """
    Generates DataLoaders for training and validation sets.

    Args:
        batch_size (int): Batch size.
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        train_loader, val_loader
    """
    set_seed(Config.SEED)

    # 1. Calculate Imputation Value (Median from Train Metadata)
    # We read the CSV directly to get the median, ensuring consistency across splits
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    angle_impute_val = df_train_meta["inc_angle"].median()

    # 2. Load and Process Data
    X_train, angles_train, y_train = _load_and_process(
        Config.TRAIN_META,
        Config.TRAIN_JSON,
        "train",
        angle_impute_val,
        load_cached_data,
    )

    X_val, angles_val, y_val = _load_and_process(
        Config.VAL_META, Config.TRAIN_JSON, "val", angle_impute_val, load_cached_data
    )

    # 3. Debug Subsetting
    if debug:
        print("DEBUG MODE: Truncating datasets to 32 samples.")
        X_train, angles_train, y_train = X_train[:32], angles_train[:32], y_train[:32]
        X_val, angles_val, y_val = X_val[:32], angles_val[:32], y_val[:32]

    # 4. Define Transforms
    # Augmentation for training only
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No augmentation for validation
    val_transform = None

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=train_transform, is_test=False
    )
    val_dataset = IcebergDataset(
        X_val, angles_val, y_val, transform=val_transform, is_test=False
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader


def get_test_loader(
    batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=True
):
    """
    Generates DataLoader for the test set.

    Args:
        batch_size (int): Batch size.
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        test_loader
    """
    set_seed(Config.SEED)

    # 1. Calculate Imputation Value (Must use Train Median for consistency)
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    angle_impute_val = df_train_meta["inc_angle"].median()

    # 2. Load and Process Data
    X_test, angles_test, ids_test = _load_and_process(
        Config.TEST_META, Config.TEST_JSON, "test", angle_impute_val, load_cached_data
    )

    # 3. Debug Subsetting
    if debug:
        print("DEBUG MODE: Truncating test dataset to 32 samples.")
        X_test, angles_test, ids_test = X_test[:32], angles_test[:32], ids_test[:32]

    # 4. Create Dataset (No transforms for test)
    test_dataset = IcebergDataset(
        X_test, angles_test, ids_test, transform=None, is_test=True
    )

    # 5. Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return test_loader
