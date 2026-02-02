import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_or_process_cache


def process_data():
    """
    Reads raw JSON data and metadata, processes images and features,
    and returns a dictionary of numpy arrays for caching.
    """
    # 1. Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_CSV)
    df_val_meta = pd.read_csv(Config.VAL_META_CSV)
    df_test_meta = pd.read_csv(Config.TEST_META_CSV)

    # Create sets/maps for O(1) lookup
    train_ids = set(df_train_meta["id"].values)
    val_ids = set(df_val_meta["id"].values)
    # Map test IDs to their desired order index
    test_id_map = {id_: i for i, id_ in enumerate(df_test_meta["id"].values)}

    # 2. Load Raw JSON Data
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)

    with open(Config.TEST_JSON, "r") as f:
        raw_test_data = json.load(f)

    # 3. Helper function to extract and format data
    def extract_data(raw_list, is_test=False):
        ids = []
        bands_1 = []
        bands_2 = []
        inc_angles = []
        labels = []

        for item in raw_list:
            ids.append(item["id"])

            # Reshape flattened 5625 list to 75x75
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
            bands_1.append(b1)
            bands_2.append(b2)

            inc_angles.append(item["inc_angle"])

            if not is_test:
                labels.append(item["is_iceberg"])

        # Stack into (N, 75, 75)
        b1_stack = np.stack(bands_1)
        b2_stack = np.stack(bands_2)

        # Construct 3rd channel: Mean of Band 1 and Band 2
        b3_stack = (b1_stack + b2_stack) / 2.0

        # Stack channels to (N, 3, 75, 75)
        X = np.stack([b1_stack, b2_stack, b3_stack], axis=1)

        # Process inc_angle: 'na' becomes NaN
        inc_angles = pd.to_numeric(inc_angles, errors="coerce")
        inc_angles = np.array(inc_angles, dtype=np.float32)

        if is_test:
            return ids, X, inc_angles, None
        else:
            return ids, X, inc_angles, np.array(labels, dtype=np.float32)

    # Extract features
    train_full_ids, X_full, inc_full, y_full = extract_data(
        raw_train_data, is_test=False
    )
    test_ids_raw, X_test_raw, inc_test_raw, _ = extract_data(
        raw_test_data, is_test=True
    )

    # 4. Imputation (Incidence Angle)
    # Identify indices belonging to the training set to compute statistics
    train_indices_in_full = [
        i for i, id_ in enumerate(train_full_ids) if id_ in train_ids
    ]

    # Compute median from Training Set ONLY
    train_inc_values = inc_full[train_indices_in_full]
    inc_median = np.nanmedian(train_inc_values)

    # Fill NaNs
    inc_full[np.isnan(inc_full)] = inc_median
    inc_test_raw[np.isnan(inc_test_raw)] = inc_median

    # 5. Min-Max Scaling
    # Compute Min/Max from Training Set ONLY
    X_train_subset = X_full[train_indices_in_full]
    g_min = X_train_subset.min()
    g_max = X_train_subset.max()

    # Apply scaling to all data
    # (x - min) / (max - min)
    X_full = (X_full - g_min) / (g_max - g_min)
    X_test_raw = (X_test_raw - g_min) / (g_max - g_min)

    # Clip to ensure [0, 1] range (handling potential outliers in test)
    X_full = np.clip(X_full, 0.0, 1.0)
    X_test_raw = np.clip(X_test_raw, 0.0, 1.0)

    # 6. Split and Organize Data
    # Pre-allocate arrays
    n_train = len(train_ids)
    n_val = len(val_ids)
    n_test = len(test_id_map)

    # Train
    X_train = np.zeros((n_train, 3, 75, 75), dtype=np.float32)
    y_train = np.zeros((n_train,), dtype=np.float32)
    meta_train = np.zeros((n_train,), dtype=np.float32)

    # Val
    X_val = np.zeros((n_val, 3, 75, 75), dtype=np.float32)
    y_val = np.zeros((n_val,), dtype=np.float32)
    meta_val = np.zeros((n_val,), dtype=np.float32)

    # Fill Train/Val based on ID matching
    train_ptr = 0
    val_ptr = 0

    for i, id_ in enumerate(train_full_ids):
        if id_ in train_ids:
            X_train[train_ptr] = X_full[i]
            y_train[train_ptr] = y_full[i]
            meta_train[train_ptr] = inc_full[i]
            train_ptr += 1
        elif id_ in val_ids:
            X_val[val_ptr] = X_full[i]
            y_val[val_ptr] = y_full[i]
            meta_val[val_ptr] = inc_full[i]
            val_ptr += 1

    # Reorder Test Data to match metadata/submission file order
    X_test = np.zeros((n_test, 3, 75, 75), dtype=np.float32)
    meta_test = np.zeros((n_test,), dtype=np.float32)
    ids_test_ordered = df_test_meta["id"].values

    for i, id_ in enumerate(test_ids_raw):
        if id_ in test_id_map:
            idx = test_id_map[id_]
            X_test[idx] = X_test_raw[i]
            meta_test[idx] = inc_test_raw[i]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "meta_test": meta_test,
        "ids_test": ids_test_ordered,
    }


class TrainTransform:
    """
    Applies random 90-degree rotations and horizontal flips using native tensor operations.
    """

    def __call__(self, x):
        # x shape: (3, 75, 75)

        # 1. Random Rotation (0, 90, 180, 270 degrees)
        # k is the number of times to rotate by 90 degrees
        k = np.random.randint(0, 4)
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])

        # 2. Random Horizontal Flip
        if np.random.random() > 0.5:
            x = torch.flip(x, dims=[2])

        return x


class IcebergDataset(Dataset):
    def __init__(self, X, meta, y=None, transform=None):
        self.X = X
        self.meta = meta
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy to tensor
        img = torch.from_numpy(self.X[idx])
        meta = torch.tensor([self.meta[idx]], dtype=torch.float32)  # Shape (1,)

        # Apply augmentations
        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)  # Shape (1,)
            return img, meta, label
        else:
            return img, meta


def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Load or process data
    data = load_or_process_cache(
        Config.PROCESSED_DATA_CACHE, process_data, load_cache=load_cached_data
    )

    # Handle Debug Mode
    if Config.DEBUG:
        limit = Config.DEBUG_SAMPLE_SIZE
        data["X_train"] = data["X_train"][:limit]
        data["y_train"] = data["y_train"][:limit]
        data["meta_train"] = data["meta_train"][:limit]

        data["X_val"] = data["X_val"][:limit]
        data["y_val"] = data["y_val"][:limit]
        data["meta_val"] = data["meta_val"][:limit]

        data["X_test"] = data["X_test"][:limit]
        data["meta_test"] = data["meta_test"][:limit]
        data["ids_test"] = data["ids_test"][:limit]

    # Initialize Datasets
    # Apply augmentation only to training set
    train_ds = IcebergDataset(
        data["X_train"], data["meta_train"], data["y_train"], transform=TrainTransform()
    )

    val_ds = IcebergDataset(data["X_val"], data["meta_val"], data["y_val"])

    test_ds = IcebergDataset(data["X_test"], data["meta_test"])

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, data["ids_test"]
