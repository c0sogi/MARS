import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        PyTorch Dataset for Iceberg/Ship classification.

        Args:
            images (np.ndarray): Image data of shape (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            labels (np.ndarray, optional): Target labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Transformations to apply to the image tensor.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and convert to Tensor (C, H, W)
        # Input numpy array is (75, 75, 3), output tensor should be (3, 75, 75)
        img_np = self.images[idx]
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1)

        # Apply augmentations if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Convert angle to tensor
        angle_tensor = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.labels is not None:
            # Training/Validation mode: Return label
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # Inference mode: Return ID
            img_id = self.ids[idx] if self.ids is not None else ""
            return img_tensor, angle_tensor, img_id


def _process_data(json_path, cache_prefix, load_cached_data):
    """
    Internal helper to load raw JSON data, process it into numpy arrays, and cache it.

    Args:
        json_path (str): Path to the raw JSON file.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, y, angles, ids) as numpy arrays. y is None for test set.
    """
    # Define cache file paths
    cache_x = os.path.join(Config.WORKING_DIR, f"X_{cache_prefix}.npy")
    cache_y = os.path.join(Config.WORKING_DIR, f"y_{cache_prefix}.npy")
    cache_a = os.path.join(Config.WORKING_DIR, f"angle_{cache_prefix}.npy")
    cache_i = os.path.join(Config.WORKING_DIR, f"ids_{cache_prefix}.npy")

    # Check if all required cache files exist
    has_cache = (
        os.path.exists(cache_x) and os.path.exists(cache_a) and os.path.exists(cache_i)
    )
    # For training data, labels must also be cached
    if "train" in cache_prefix:
        has_cache = has_cache and os.path.exists(cache_y)

    if load_cached_data and has_cache:
        print(f"Loading cached {cache_prefix} data from {Config.WORKING_DIR}...")
        X = np.load(cache_x)
        angles = np.load(cache_a)
        ids = np.load(cache_i, allow_pickle=True)
        y = np.load(cache_y) if "train" in cache_prefix else None
        return X, y, angles, ids

    # If not cached or reload requested, process from scratch
    print(f"Processing raw data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images: Band 1 (HH), Band 2 (HV), Band 3 (Avg)
    # Each band in JSON is a flattened list of 5625 floats. Reshape to 75x75.
    b1_list = [np.array(b, dtype=np.float32).reshape(75, 75) for b in df["band_1"]]
    b2_list = [np.array(b, dtype=np.float32).reshape(75, 75) for b in df["band_2"]]

    b1 = np.stack(b1_list)  # Shape: (N, 75, 75)
    b2 = np.stack(b2_list)  # Shape: (N, 75, 75)
    b3 = (b1 + b2) / 2.0  # Synthetic 3rd band

    # Stack channels last: (N, 75, 75, 3)
    X = np.stack([b1, b2, b3], axis=-1)

    # Process Angles: Convert 'na' to NaN and ensure float32
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # Process IDs
    ids = df["id"].values

    # Process Labels (only if present)
    y = None
    if "is_iceberg" in df.columns:
        y = df["is_iceberg"].values.astype(np.float32)

    # Save processed data to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_a, angles)
    np.save(cache_i, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, angles, ids


def get_train_val_loaders(fold_index, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold of the training set.
    Implements 'Leak-Free Preprocessing' by imputing missing angles using statistics
    derived solely from the training split of the current fold.

    Args:
        fold_index (int): The fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load the entire training dataset
    X, y, angles, ids = _process_data(Config.TRAIN_JSON, "train", load_cached_data)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Generate splits and select the requested fold
    splits = list(skf.split(X, y))
    if fold_index < 0 or fold_index >= Config.N_FOLDS:
        raise ValueError(
            f"Fold index {fold_index} is out of range (0-{Config.N_FOLDS-1})"
        )

    train_idx, val_idx = splits[fold_index]

    # Split the data arrays
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ang_train, ang_val = angles[train_idx], angles[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Leak-Free Imputation: Calculate median angle ONLY on training data
    median_angle = np.nanmedian(ang_train)

    # Apply imputation to copies of the arrays
    ang_train_imp = ang_train.copy()
    ang_train_imp[np.isnan(ang_train_imp)] = median_angle

    ang_val_imp = ang_val.copy()
    ang_val_imp[np.isnan(ang_val_imp)] = median_angle

    # Define Transforms
    # Training: Random flips for regularization
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )
    # Validation: No augmentation

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train_imp, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val_imp, y_val, ids_val, transform=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates a DataLoader for the test set.
    Imputes missing angles using the global median from the training set.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        DataLoader: The test data loader.
    """
    # Load test data
    X_test, _, ang_test, ids_test = _process_data(
        Config.TEST_JSON, "test", load_cached_data
    )

    # Load training data solely to calculate the global median angle
    _, _, ang_train, _ = _process_data(Config.TRAIN_JSON, "train", load_cached_data)
    median_angle = np.nanmedian(ang_train)

    # Impute missing angles in test set
    ang_test_imp = ang_test.copy()
    ang_test_imp[np.isnan(ang_test_imp)] = median_angle

    # Create Dataset (No labels, No transforms)
    test_dataset = IcebergDataset(
        X_test, ang_test_imp, labels=None, ids=ids_test, transform=None
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
