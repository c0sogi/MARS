import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel images, incidence angles, and labels.
    """

    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.array): Scaled images of shape (N, 75, 75, 3).
            angles (np.array): Imputed incidence angles of shape (N,).
            labels (np.array, optional): Labels of shape (N,).
            transform (bool): Whether to apply geometric augmentations.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        angle = self.angles[idx]

        if self.transform:
            # Random rotation: 0, 90, 180, 270 degrees
            # k is number of 90 degree rotations
            k = np.random.randint(0, 4)
            image = np.rot90(image, k=k)

            # Random horizontal flip
            if np.random.random() < 0.5:
                image = np.fliplr(image)

        # Convert to tensor: (H, W, C) -> (C, H, W)
        # .copy() is required because rot90/flip can create negative strides which torch doesn't support
        image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor
        else:
            return image_tensor, angle_tensor


def load_raw_data(load_cached_data=True):
    """
    Loads raw data from JSON files or Cache.
    Constructs 3-channel images (Band1, Band2, Mean).
    Returns unscaled numpy arrays.
    """
    cache_path = Config.CACHE_PATH

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading data from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            return {
                "X_train": data["X_train"],
                "y_train": data["y_train"],
                "angles_train": data["angles_train"],
                "ids_train": data["ids_train"],
                "X_test": data["X_test"],
                "angles_test": data["angles_test"],
                "ids_test": data["ids_test"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process from scratch
    print("Processing raw JSON data...")

    # Load Metadata to identify splits and IDs
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Combine train and val metadata to get the full labeled dataset
    df_full_train = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(
        drop=True
    )
    train_ids_set = set(df_full_train["id"].values)
    test_ids_set = set(df_test_meta["id"].values)

    # Load JSON files
    with open(Config.TRAIN_JSON, "r") as f:
        train_json = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_json = json.load(f)

    def process_json_list(data_list, target_ids, has_labels=True):
        images = []
        angles = []
        labels = []
        ids = []

        for item in data_list:
            if item["id"] not in target_ids:
                continue

            # Extract Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Construct 3rd channel: Mean
            b3 = (b1 + b2) / 2.0

            # Stack to (75, 75, 3)
            img = np.dstack((b1, b2, b3))
            images.append(img)

            # Extract Angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            ids.append(item["id"])

            if has_labels:
                labels.append(item["is_iceberg"])

        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        ang = np.array(angles, dtype=np.float32)
        ids = np.array(ids)

        if has_labels:
            y = np.array(labels, dtype=np.int32)
            return X, ang, y, ids
        else:
            return X, ang, ids

    # Process Train
    X_train, angles_train, y_train, ids_train = process_json_list(
        train_json, train_ids_set, has_labels=True
    )

    # Process Test
    X_test, angles_test, ids_test = process_json_list(
        test_json, test_ids_set, has_labels=False
    )

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        angles_train=angles_train,
        ids_train=ids_train,
        X_test=X_test,
        angles_test=angles_test,
        ids_test=ids_test,
    )

    print(f"Data processed and cached to {cache_path}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angles_train": angles_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angles_test": angles_test,
        "ids_test": ids_test,
    }


def get_fold_loaders(fold, load_cached_data=True):
    """
    Prepares DataLoaders for a specific fold with strict leakage prevention.

    Args:
        fold (int): The fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader: DataLoader for training (augmented).
        val_loader: DataLoader for validation (not augmented).
        scaling_stats: List of (min, max) tuples for the 3 channels.
        angle_mean: Mean incidence angle from the training set (for imputation).
    """
    # 1. Load Data
    data = load_raw_data(load_cached_data)
    X = data["X_train"]
    y = data["y_train"]
    angles = data["angles_train"]

    # Debugging: Subset data if configured
    if Config.DEBUG:
        print("DEBUG MODE: Subsetting data...")
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(X))
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]
        angles = angles[indices]
        n_splits = 2  # Force 2 folds for debug
    else:
        n_splits = Config.NUM_FOLDS

    # 2. Stratified Split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)

    # Iterate to get the specific fold indices
    fold_generator = skf.split(X, y)
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(fold_generator):
        if i == fold:
            train_idx, val_idx = t_idx, v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold {fold} out of range for {n_splits} splits.")

    # 3. Split Data
    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ang_train_raw, ang_val_raw = angles[train_idx], angles[val_idx]

    # 4. Strict Fold-wise Preprocessing

    # A. Impute Incidence Angle
    # Compute mean ONLY on training data
    angle_mean = np.nanmean(ang_train_raw)

    # Apply imputation
    ang_train = np.where(np.isnan(ang_train_raw), angle_mean, ang_train_raw)
    ang_val = np.where(np.isnan(ang_val_raw), angle_mean, ang_val_raw)

    # B. Min-Max Scaling
    # Compute stats ONLY on training data
    scaling_stats = []
    X_train = np.zeros_like(X_train_raw)
    X_val = np.zeros_like(X_val_raw)

    for c in range(3):  # For each channel
        c_data = X_train_raw[:, :, :, c]
        _min = c_data.min()
        _max = c_data.max()

        denom = _max - _min
        if denom == 0:
            denom = 1.0

        scaling_stats.append((_min, _max))

        # Apply scaling
        X_train[:, :, :, c] = (X_train_raw[:, :, :, c] - _min) / denom
        X_val[:, :, :, c] = (X_val_raw[:, :, :, c] - _min) / denom

    # 5. Create Datasets
    train_dataset = IcebergDataset(X_train, ang_train, y_train, transform=True)
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=False)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, scaling_stats, angle_mean


def get_test_loader(scaling_stats, angle_mean, load_cached_data=True):
    """
    Prepares DataLoader for the test set using provided scaling stats.

    Args:
        scaling_stats: List of (min, max) tuples for the 3 channels.
        angle_mean: Float value to replace NaNs in incidence angle.

    Returns:
        loader: DataLoader for test set.
        ids: Array of test image IDs.
    """
    # 1. Load Data
    data = load_raw_data(load_cached_data)
    X_test_raw = data["X_test"]
    angles_test_raw = data["angles_test"]
    ids_test = data["ids_test"]

    # 2. Impute Incidence Angle
    angles_test = np.where(np.isnan(angles_test_raw), angle_mean, angles_test_raw)

    # 3. Apply Scaling
    X_test = np.zeros_like(X_test_raw)
    for c in range(3):
        _min, _max = scaling_stats[c]
        denom = _max - _min
        if denom == 0:
            denom = 1.0

        X_test[:, :, :, c] = (X_test_raw[:, :, :, c] - _min) / denom

    # 4. Create Dataset and Loader
    dataset = IcebergDataset(X_test, angles_test, labels=None, transform=False)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader, ids_test
