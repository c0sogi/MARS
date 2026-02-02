import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles conversion to Tensor and on-the-fly augmentation.
    """

    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to FloatTensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply augmentations if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into 3-channel images, imputes missing angles,
    and caches the result as numpy arrays.

    Returns:
        tuple: (X_train, y_train, angles_train, X_test, ids_test, angles_test)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "angles_train": os.path.join(cache_dir, "angles_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
        "angles_test": os.path.join(cache_dir, "angles_test.npy"),
    }

    # Check if cache exists
    all_cached = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        angles_train = np.load(files["angles_train"])
        X_test = np.load(files["X_test"])
        ids_test = np.load(files["ids_test"])
        angles_test = np.load(files["angles_test"])
    else:
        print("Processing data from scratch...")

        # 1. Load Metadata
        # Combine train and val metadata to get the full labeled dataset for CV
        train_meta = pd.read_csv(Config.TRAIN_META_PATH)
        val_meta = pd.read_csv(Config.VAL_META_PATH)
        full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

        test_meta = pd.read_csv(Config.TEST_META_PATH)

        # 2. Load Raw JSON Data
        with open(Config.TRAIN_JSON, "r") as f:
            raw_train = json.load(f)
        with open(Config.TEST_JSON, "r") as f:
            raw_test = json.load(f)

        # Create dictionaries for fast lookup by ID
        train_dict = {item["id"]: item for item in raw_train}
        test_dict = {item["id"]: item for item in raw_test}

        # Helper function to process samples
        def process_samples(meta_df, data_dict, is_train=True):
            X_list = []
            angles_list = []
            y_list = []
            ids_list = []

            for _, row in meta_df.iterrows():
                img_id = row["id"]
                item = data_dict[img_id]

                # Extract Bands
                b1 = np.array(item["band_1"]).reshape(75, 75)
                b2 = np.array(item["band_2"]).reshape(75, 75)

                # Create 3rd channel (Average)
                b3 = (b1 + b2) / 2.0

                # Stack to (3, 75, 75)
                img = np.stack([b1, b2, b3], axis=0)
                X_list.append(img)

                # Extract Angle
                ang = item["inc_angle"]
                if ang == "na":
                    angles_list.append(np.nan)
                else:
                    angles_list.append(float(ang))

                ids_list.append(img_id)

                if is_train:
                    y_list.append(item["is_iceberg"])

            X = np.array(X_list, dtype=np.float32)
            angles = np.array(angles_list, dtype=np.float32)
            ids = np.array(ids_list)

            if is_train:
                y = np.array(y_list, dtype=np.float32)
                return X, angles, y, ids
            else:
                return X, angles, ids

        # Process Datasets
        X_train, angles_train, y_train, _ = process_samples(
            full_train_meta, train_dict, is_train=True
        )
        X_test, angles_test, ids_test = process_samples(
            test_meta, test_dict, is_train=False
        )

        # 3. Impute Missing Angles
        # Compute median from valid training angles
        angle_median = np.nanmedian(angles_train)

        # Fill NaNs
        angles_train[np.isnan(angles_train)] = angle_median
        angles_test[np.isnan(angles_test)] = angle_median

        # Save to cache
        print(f"Saving processed data to {cache_dir}...")
        np.save(files["X_train"], X_train)
        np.save(files["y_train"], y_train)
        np.save(files["angles_train"], angles_train)
        np.save(files["X_test"], X_test)
        np.save(files["ids_test"], ids_test)
        np.save(files["angles_test"], angles_test)

    # Handle Debugging / Subsampling
    if Config.MAX_SAMPLES is not None:
        print(f"DEBUG: Limiting dataset to {Config.MAX_SAMPLES} samples.")
        X_train = X_train[: Config.MAX_SAMPLES]
        y_train = y_train[: Config.MAX_SAMPLES]
        angles_train = angles_train[: Config.MAX_SAMPLES]
        X_test = X_test[: Config.MAX_SAMPLES]
        ids_test = ids_test[: Config.MAX_SAMPLES]
        angles_test = angles_test[: Config.MAX_SAMPLES]

    return X_train, y_train, angles_train, X_test, ids_test, angles_test


def get_cv_loaders(fold_idx, X, y, angles):
    """
    Creates train and validation DataLoaders for a specific fold using StratifiedKFold.

    Args:
        fold_idx (int): The current fold index (0 to NUM_FOLDS-1).
        X (np.ndarray): Training images.
        y (np.ndarray): Training labels.
        angles (np.ndarray): Training incidence angles.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get split indices
    # skf.split requires X and y. We provide zeros for X as only y matters for stratification.
    splits = list(skf.split(np.zeros(len(y)), y))

    if fold_idx < 0 or fold_idx >= Config.NUM_FOLDS:
        raise ValueError(f"Fold index {fold_idx} out of range (0-{Config.NUM_FOLDS-1})")

    train_idx, val_idx = splits[fold_idx]

    # Create subsets
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    angles_train_fold = angles[train_idx]

    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]
    angles_val_fold = angles[val_idx]

    # Define Augmentations for Training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train_fold, angles_train_fold, y_train_fold, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val_fold, angles_val_fold, y_val_fold, transform=None
    )

    # Create DataLoaders
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

    return train_loader, val_loader


def get_test_loader(X, angles, ids):
    """
    Creates a DataLoader for the test set.
    """
    dataset = IcebergDataset(X, angles, y=None, transform=None)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
