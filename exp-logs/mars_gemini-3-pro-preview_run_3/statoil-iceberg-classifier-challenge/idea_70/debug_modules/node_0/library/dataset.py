import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    def __init__(self, X, angles, ids, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            ids (np.ndarray): Image IDs of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Augmentation transform.
        """
        self.X = X
        self.angles = angles
        self.ids = ids
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (3, 75, 75) float32
        img = self.X[idx]
        angle = self.angles[idx]
        id_ = self.ids[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img)

        # Apply transforms (Augmentation)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.y is not None:
            label_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor, id_


def get_transforms(mode="train"):
    """
    Returns the torchvision transforms for the given mode.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # No test-time augmentation
        return None


def preprocess_data(load_cached_data=True):
    """
    Loads raw data, processes it into numpy arrays, and caches it.
    Returns the raw arrays (with NaNs in angles) for downstream splitting.

    Returns:
        tuple: (X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Cache filenames
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "angles_train": os.path.join(cache_dir, "angle_train.npy"),
        "ids_train": os.path.join(cache_dir, "ids_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "angles_test": os.path.join(cache_dir, "angle_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(f) for f in files.values()):
        print("Loading data from cache...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        angles_train = np.load(files["angles_train"])
        ids_train = np.load(files["ids_train"])
        X_test = np.load(files["X_test"])
        angles_test = np.load(files["angles_test"])
        ids_test = np.load(files["ids_test"])
        return X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test

    print("Processing raw data from JSON...")

    # Helper to process image bands
    def process_images(df):
        # Extract bands and reshape to 75x75
        b1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        b2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )
        # Channel 3: Average of Band 1 and Band 2
        b3 = (b1 + b2) / 2.0
        # Stack: (N, 3, 75, 75)
        X = np.stack([b1, b2, b3], axis=1)
        return X

    # Load Train Data
    # We load the full dataset to allow for 5-Fold CV later.
    df_train = pd.read_json(Config.TRAIN_JSON)
    X_train = process_images(df_train)
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    ids_train = df_train["id"].values
    # Convert angles to numeric, coercing 'na' to NaN
    angles_train = pd.to_numeric(df_train["inc_angle"], errors="coerce").values.astype(
        np.float32
    )

    # Load Test Data
    df_test = pd.read_json(Config.TEST_JSON)
    X_test = process_images(df_test)
    ids_test = df_test["id"].values
    angles_test = pd.to_numeric(df_test["inc_angle"], errors="coerce").values.astype(
        np.float32
    )

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["angles_train"], angles_train)
    np.save(files["ids_train"], ids_train)
    np.save(files["X_test"], X_test)
    np.save(files["angles_test"], angles_test)
    np.save(files["ids_test"], ids_test)

    return X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test


def get_fold_datasets(X, y, angles, ids, fold, num_folds=5, seed=Config.SEED):
    """
    Splits data into train/val for a specific fold, performs leak-free imputation
    of incidence angles, and returns PyTorch Datasets.

    Args:
        X, y, angles, ids: Raw data arrays (angles may contain NaNs).
        fold (int): Current fold index (0 to num_folds-1).
        num_folds (int): Total number of folds.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Stratified K-Fold for class balance
    # We cast y to int for stratification (though it works with float 0.0/1.0 usually)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Get indices for the requested fold
    splits = list(skf.split(X, y.astype(int)))
    train_idx, val_idx = splits[fold]

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angles_train, angles_val = angles[train_idx], angles[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Leak-free Imputation: Calculate median ONLY on training data
    train_median_angle = np.nanmedian(angles_train)

    # Fill NaNs
    angles_train_filled = np.where(
        np.isnan(angles_train), train_median_angle, angles_train
    )
    angles_val_filled = np.where(np.isnan(angles_val), train_median_angle, angles_val)

    # Create Datasets
    train_ds = IcebergDataset(
        X_train,
        angles_train_filled,
        ids_train,
        y_train,
        transform=get_transforms("train"),
    )
    val_ds = IcebergDataset(
        X_val, angles_val_filled, ids_val, y_val, transform=get_transforms("val")
    )

    return train_ds, val_ds


def get_test_dataset(X_test, angles_test, ids_test, angles_train_all):
    """
    Creates test dataset with imputation based on the full training set median.
    """
    # Impute test angles using global training median
    train_median_angle = np.nanmedian(angles_train_all)
    angles_test_filled = np.where(
        np.isnan(angles_test), train_median_angle, angles_test
    )

    test_ds = IcebergDataset(
        X_test, angles_test_filled, ids_test, y=None, transform=get_transforms("test")
    )
    return test_ds
