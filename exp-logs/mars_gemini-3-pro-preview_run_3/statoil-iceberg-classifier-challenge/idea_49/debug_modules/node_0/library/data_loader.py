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


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,). Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and convert to tensor
        # Images are already (3, 75, 75) float32
        image = torch.from_numpy(self.images[idx])

        # Load angle
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply transforms if any (Augmentation)
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(
                self.labels[idx], dtype=torch.float32
            )  # BCEWithLogitsLoss expects float target
            return image, angle, label
        else:
            return image, angle


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes bands into images, handles missing angles,
    and caches the result as numpy arrays.

    Returns:
        tuple: (X_train_all, y_train_all, ang_train_all, X_test, ang_test, test_ids)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "ang_train": os.path.join(cache_dir, "ang_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "ang_test": os.path.join(cache_dir, "ang_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check if cache exists
    if load_cached_data and all(os.path.exists(f) for f in files.values()):
        print("Loading data from cache...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        ang_train = np.load(files["ang_train"])
        X_test = np.load(files["X_test"])
        ang_test = np.load(files["ang_test"])
        ids_test = np.load(files["ids_test"], allow_pickle=True)
        return X_train, y_train, ang_train, X_test, ang_test, ids_test

    print("Processing raw data from JSON files...")

    # 1. Load Metadata to identify IDs (though we load full json, this ensures we follow structure)
    # We combine train and val metadata to get the full labeled set for CV
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # 2. Load Raw JSONs
    # Reading train.json
    with open(Config.TRAIN_JSON, "r") as f:
        train_data_raw = json.load(f)
    df_train = pd.DataFrame(train_data_raw)

    # Reading test.json
    with open(Config.TEST_JSON, "r") as f:
        test_data_raw = json.load(f)
    df_test = pd.DataFrame(test_data_raw)

    # 3. Helper function to process images
    def process_images(df):
        x_band1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        x_band2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )

        # Channel 3: Average of Band 1 and Band 2
        x_band3 = (x_band1 + x_band2) / 2.0

        # Stack to (N, 3, 75, 75)
        # np.stack creates (N, 75, 75, 3) if axis=-1, or (N, 3, 75, 75) if axis=1
        # PyTorch prefers (C, H, W)
        X = np.stack((x_band1, x_band2, x_band3), axis=1)
        return X

    # 4. Helper function to process angles
    def process_angles(df, median_val=None):
        # Coerce 'na' to NaN
        angles = pd.to_numeric(df["inc_angle"], errors="coerce")
        if median_val is None:
            median_val = angles.median()

        # Fill NaN with median
        angles = angles.fillna(median_val).astype(np.float32).values
        return angles, median_val

    # Process Training Data
    X_train = process_images(df_train)
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    ang_train, median_angle = process_angles(df_train)

    # Process Test Data
    X_test = process_images(df_test)
    ang_test, _ = process_angles(df_test, median_val=median_angle)
    ids_test = df_test["id"].values

    # Save to cache
    print("Saving processed data to cache...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["ang_train"], ang_train)
    np.save(files["X_test"], X_test)
    np.save(files["ang_test"], ang_test)
    np.save(files["ids_test"], ids_test)

    return X_train, y_train, ang_train, X_test, ang_test, ids_test


def get_loaders(fold=0, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in Stratified K-Fold Cross Validation.

    Args:
        fold (int): The fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(Config.SEED)

    # Load all data
    X_all, y_all, ang_all, X_test, ang_test, ids_test = process_data(load_cached_data)

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # skf.split returns a generator, we iterate to find the specific fold
    splits = list(skf.split(X_all, y_all))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for {Config.NUM_FOLDS} splits.")

    train_idx, val_idx = splits[fold]

    # Subset data
    X_train_fold = X_all[train_idx]
    y_train_fold = y_all[train_idx]
    ang_train_fold = ang_all[train_idx]

    X_val_fold = X_all[val_idx]
    y_val_fold = y_all[val_idx]
    ang_val_fold = ang_all[val_idx]

    # Define Transforms
    # Simple flips for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # No transforms for validation/test
    val_transform = None

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train_fold, ang_train_fold, y_train_fold, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val_fold, ang_val_fold, y_val_fold, transform=val_transform
    )
    test_dataset = IcebergDataset(X_test, ang_test, labels=None, transform=None)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
