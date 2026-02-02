import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg Detection.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,).
            ids (np.ndarray, optional): Shape (N,).
            transform (callable, optional): Augmentations.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (3, 75, 75)
        angle = self.angles[idx]  # Scalar

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply augmentations
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Return tuple based on mode (Train/Val vs Test)
        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor
        else:
            # Test mode: return ID for submission generation
            img_id = self.ids[idx]
            return image_tensor, angle_tensor, img_id


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays (images, angles, labels),
    imputes missing values, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (train_data_dict, test_data_dict)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # File paths for caching
    f_train_X = os.path.join(cache_dir, "X_train.npy")
    f_train_y = os.path.join(cache_dir, "y_train.npy")
    f_train_a = os.path.join(cache_dir, "angles_train.npy")
    f_train_i = os.path.join(cache_dir, "ids_train.npy")

    f_test_X = os.path.join(cache_dir, "X_test.npy")
    f_test_a = os.path.join(cache_dir, "angles_test.npy")
    f_test_i = os.path.join(cache_dir, "ids_test.npy")

    # Check existence
    train_exists = all(
        os.path.exists(f) for f in [f_train_X, f_train_y, f_train_a, f_train_i]
    )
    test_exists = all(os.path.exists(f) for f in [f_test_X, f_test_a, f_test_i])

    if load_cached_data and train_exists and test_exists:
        print("Loading processed data from cache...")
        train_data = {
            "X": np.load(f_train_X),
            "y": np.load(f_train_y),
            "angles": np.load(f_train_a),
            "ids": np.load(f_train_i, allow_pickle=True),
        }
        test_data = {
            "X": np.load(f_test_X),
            "angles": np.load(f_test_a),
            "ids": np.load(f_test_i, allow_pickle=True),
        }
        return train_data, test_data

    print("Processing raw data from scratch...")

    # --- Process Training Data ---
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train = json.load(f)
    df_train = pd.DataFrame(raw_train)

    # Process Images: List -> (75, 75) -> Stack -> (N, 3, 75, 75)
    # Band 1 (HH)
    b1_train = np.array(
        [np.array(b).astype(np.float32).reshape(75, 75) for b in df_train["band_1"]]
    )
    # Band 2 (HV)
    b2_train = np.array(
        [np.array(b).astype(np.float32).reshape(75, 75) for b in df_train["band_2"]]
    )
    # Band 3 (Avg)
    b3_train = (b1_train + b2_train) / 2.0
    X_train = np.stack([b1_train, b2_train, b3_train], axis=1)

    y_train = df_train["is_iceberg"].values.astype(np.float32)
    ids_train = df_train["id"].values

    # Process Angles
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    angles_train = df_train["inc_angle"].values.astype(np.float32)

    # Imputation: Calculate median from valid training samples
    angle_median = np.nanmedian(angles_train)
    angles_train[np.isnan(angles_train)] = angle_median

    # --- Process Test Data ---
    with open(Config.TEST_JSON, "r") as f:
        raw_test = json.load(f)
    df_test = pd.DataFrame(raw_test)

    b1_test = np.array(
        [np.array(b).astype(np.float32).reshape(75, 75) for b in df_test["band_1"]]
    )
    b2_test = np.array(
        [np.array(b).astype(np.float32).reshape(75, 75) for b in df_test["band_2"]]
    )
    b3_test = (b1_test + b2_test) / 2.0
    X_test = np.stack([b1_test, b2_test, b3_test], axis=1)

    ids_test = df_test["id"].values

    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    angles_test = df_test["inc_angle"].values.astype(np.float32)

    # Imputation: Use training median
    angles_test[np.isnan(angles_test)] = angle_median

    # --- Save to Cache ---
    np.save(f_train_X, X_train)
    np.save(f_train_y, y_train)
    np.save(f_train_a, angles_train)
    np.save(f_train_i, ids_train)

    np.save(f_test_X, X_test)
    np.save(f_test_a, angles_test)
    np.save(f_test_i, ids_test)

    train_data = {"X": X_train, "y": y_train, "angles": angles_train, "ids": ids_train}
    test_data = {"X": X_test, "angles": angles_test, "ids": ids_test}

    return train_data, test_data


def get_loaders(fold_idx=0, debug=False):
    """
    Prepares DataLoaders for training, validation, and testing.
    Uses Stratified K-Fold to split the training data.

    Args:
        fold_idx (int): Current fold index (0 to NUM_FOLDS-1).
        debug (bool): If True, uses a subset of data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(Config.SEED)

    # Load Data
    train_data, test_data = process_and_cache_data(load_cached_data=True)

    X = train_data["X"]
    y = train_data["y"]
    angles = train_data["angles"]
    ids = train_data["ids"]

    # Filter to strict training set (exclude holdout)
    # Cite debug_lesson_7
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    valid_ids = set(train_meta["id"].values)
    mask = np.isin(ids, list(valid_ids))

    X = X[mask]
    y = y[mask]
    angles = angles[mask]
    ids = ids[mask]

    X_test = test_data["X"]
    angles_test = test_data["angles"]
    ids_test = test_data["ids"]

    # Debug Mode: Slice data
    if debug or Config.DEBUG:
        limit = Config.MAX_DEBUG_SAMPLES
        print(f"DEBUG MODE: Limiting data to {limit} samples.")
        X = X[:limit]
        y = y[:limit]
        angles = angles[:limit]
        ids = ids[:limit]

        X_test = X_test[:limit]
        angles_test = angles_test[:limit]
        ids_test = ids_test[:limit]

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(X, y))

    if fold_idx < 0 or fold_idx >= Config.NUM_FOLDS:
        raise ValueError(
            f"Fold index {fold_idx} is out of range (0-{Config.NUM_FOLDS-1})"
        )

    train_idx, val_idx = splits[fold_idx]

    # Create Subsets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angles_train, angles_val = angles[train_idx], angles[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, y_val, ids_val, transform=None)
    test_dataset = IcebergDataset(
        X_test, angles_test, labels=None, ids=ids_test, transform=None
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
