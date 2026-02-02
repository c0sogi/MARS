import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


def augment_image(image):
    """
    Applies random 90-degree rotations and horizontal flips to the image tensor.

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W).

    Returns:
        torch.Tensor: Augmented image.
    """
    # Random rotation: 0, 1, 2, or 3 times 90 degrees
    k = np.random.randint(0, 4)
    image = torch.rot90(image, k, dims=[1, 2])

    # Random horizontal flip
    if np.random.random() < 0.5:
        image = torch.flip(image, dims=[2])

    return image


class IcebergDataset(Dataset):
    def __init__(self, images, inc_angles, labels=None, ids=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            inc_angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            ids (list/array, optional): Image IDs
            transform (bool): Whether to apply augmentation
        """
        self.images = images
        self.inc_angles = inc_angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to tensor
        image = torch.from_numpy(self.images[idx]).float()
        inc_angle = torch.tensor(self.inc_angles[idx], dtype=torch.float32)

        # Apply augmentation if requested
        if self.transform:
            image = augment_image(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, inc_angle, label
        else:
            # For test set, return ID as well (or just image/angle depending on usage)
            # Returning ID allows mapping predictions back
            img_id = self.ids[idx] if self.ids is not None else ""
            return image, inc_angle, img_id


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it (imputation, channel creation, normalization),
    and caches the result.

    Returns:
        tuple: (X_train, y_train, inc_train, X_test, inc_test, test_ids)
    """
    cache_path = Config.CACHE_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["X_train"],
                data["y_train"],
                data["inc_train"],
                data["X_test"],
                data["inc_test"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print("Processing data from source JSON files...")

    # Load JSONs
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Handle Incidence Angles
    # Replace 'na' with NaN, then fill with mean of training set
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")

    inc_mean = df_train["inc_angle"].mean()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(inc_mean)
    df_test["inc_angle"] = df_test["inc_angle"].fillna(inc_mean)

    # Helper to reshape bands and create 3rd channel
    def get_images(df):
        b1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        b2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )

        # Channel 3: Average of Band 1 and Band 2
        b3 = (b1 + b2) / 2.0

        # Stack: (N, 3, 75, 75)
        images = np.stack([b1, b2, b3], axis=1)
        return images

    print("Constructing image tensors...")
    X_train = get_images(df_train)
    X_test = get_images(df_test)

    y_train = df_train["is_iceberg"].values.astype(np.float32)
    inc_train = df_train["inc_angle"].values.astype(np.float32)
    inc_test = df_test["inc_angle"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # Global Min-Max Scaling
    # Compute stats on Training set only
    print("Applying Global Min-Max Scaling...")
    for i in range(3):  # For each channel
        # Flatten channel i for all training images to compute global stats
        ch_data = X_train[:, i, :, :]
        _min = ch_data.min()
        _max = ch_data.max()

        # Avoid division by zero
        denom = _max - _min
        if denom == 0:
            denom = 1.0

        # Apply to Train
        X_train[:, i, :, :] = (X_train[:, i, :, :] - _min) / denom
        # Apply to Test
        X_test[:, i, :, :] = (X_test[:, i, :, :] - _min) / denom

    # Cache results
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )
    print(f"Data processed and saved to {cache_path}")

    return X_train, y_train, inc_train, X_test, inc_test, test_ids


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation loaders for a specific fold using Stratified K-Fold.

    Args:
        fold_idx (int): The fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load Data
    X, y, inc, _, _, _ = process_data(load_cached_data)

    # Debugging Subset
    if Config.DEBUG:
        subset_size = min(len(X), Config.DEBUG_SUBSET_SIZE)
        X = X[:subset_size]
        y = y[:subset_size]
        inc = inc[:subset_size]
        print(f"DEBUG MODE: Reduced dataset to {subset_size} samples.")

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # We iterate to find the specific fold indices
    splits = list(skf.split(X, y))
    train_idx, val_idx = splits[fold_idx]

    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    inc_train_fold, inc_val_fold = inc[train_idx], inc[val_idx]

    # Create Datasets
    # Train gets augmentation (transform=True)
    train_dataset = IcebergDataset(
        X_train_fold, inc_train_fold, y_train_fold, transform=True
    )

    # Val gets no augmentation (transform=False)
    val_dataset = IcebergDataset(X_val_fold, inc_val_fold, y_val_fold, transform=False)

    # Create Loaders
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

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns the test data loader.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader: Test data loader.
    """
    _, _, _, X_test, inc_test, test_ids = process_data(load_cached_data)

    test_dataset = IcebergDataset(
        X_test, inc_test, labels=None, ids=test_ids, transform=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
