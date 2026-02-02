import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    Handles 3-channel radar images (HH, HV, Avg) and incidence angles.
    """

    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 75, 75, 3).
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
        # Retrieve image
        img = self.X[idx]  # Shape: (75, 75, 3)

        # Convert to Tensor and rearrange dimensions: (H, W, C) -> (C, H, W)
        # We do not use ToTensor() from torchvision because it might attempt to scale
        # the float data. We manually permute to keep raw dB values.
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Apply augmentations if provided (e.g., Flips)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Retrieve angle
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Return with or without label
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle, label
        else:
            return img_tensor, angle


def _process_json_to_numpy(df, is_test=False):
    """
    Helper to convert DataFrame with flattened bands into shaped numpy arrays.
    """
    # Extract bands: list of lists -> numpy array -> reshape
    b1 = np.array(df["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(df["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    # Calculate synthetic 3rd channel: average of HH and HV
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 75, 75, 3) along the last axis
    X = np.stack([b1, b2, b3], axis=-1)

    # Process angles: coerce 'na' to NaN, convert to float
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    ids = df["id"].values

    if not is_test:
        y = df["is_iceberg"].values.astype(np.float32)
        return X, angles, y, ids
    else:
        return X, angles, None, ids


def _load_and_process_data(load_cached_data=True):
    """
    Loads data from cache or processes raw JSONs/CSVs.
    Implements median imputation for incidence angles.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "angle_train": os.path.join(cache_dir, "angle_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "angle_val": os.path.join(cache_dir, "angle_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "angle_test": os.path.join(cache_dir, "angle_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check cache existence
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading pre-processed data from cache...")
        data = {}
        for k, v in files.items():
            data[k] = np.load(v, allow_pickle=True)
            if k == "ids_test":
                data[k] = data[k].astype(str)  # Ensure IDs are strings

        return (
            (data["X_train"], data["angle_train"], data["y_train"]),
            (data["X_val"], data["angle_val"], data["y_val"]),
            (data["X_test"], data["angle_test"], data["ids_test"]),
        )

    print("Processing raw data from scratch...")

    # Load Metadata
    meta_train = pd.read_csv(Config.TRAIN_META)
    meta_val = pd.read_csv(Config.VAL_META)
    meta_test = pd.read_csv(Config.TEST_META)

    # Load Raw JSON Data
    # Reading entire JSONs into memory (fits in 220GB RAM)
    df_train_raw = pd.read_json(Config.TRAIN_JSON)
    df_test_raw = pd.read_json(Config.TEST_JSON)

    # Merge metadata with raw data to get bands for specific splits
    # We use inner join on 'id' to filter the raw data according to the split
    train_subset = pd.merge(meta_train, df_train_raw, on="id", suffixes=("", "_raw"))
    val_subset = pd.merge(meta_val, df_train_raw, on="id", suffixes=("", "_raw"))
    test_subset = pd.merge(meta_test, df_test_raw, on="id", suffixes=("", "_raw"))

    # Process subsets into numpy arrays
    X_train, ang_train, y_train, _ = _process_json_to_numpy(train_subset, is_test=False)
    X_val, ang_val, y_val, _ = _process_json_to_numpy(val_subset, is_test=False)
    X_test, ang_test, _, ids_test = _process_json_to_numpy(test_subset, is_test=True)

    # Impute Missing Incidence Angles
    # Strategy: Compute median from TRAIN set only, apply to all sets
    angle_median = np.nanmedian(ang_train)

    ang_train = np.where(np.isnan(ang_train), angle_median, ang_train)
    ang_val = np.where(np.isnan(ang_val), angle_median, ang_val)
    ang_test = np.where(np.isnan(ang_test), angle_median, ang_test)

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["X_train"], X_train)
    np.save(files["angle_train"], ang_train)
    np.save(files["y_train"], y_train)

    np.save(files["X_val"], X_val)
    np.save(files["angle_val"], ang_val)
    np.save(files["y_val"], y_val)

    np.save(files["X_test"], X_test)
    np.save(files["angle_test"], ang_test)
    np.save(files["ids_test"], ids_test)

    return (
        (X_train, ang_train, y_train),
        (X_val, ang_val, y_val),
        (X_test, ang_test, ids_test),
    )


def get_loaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load data
    (
        (X_train, ang_train, y_train),
        (X_val, ang_val, y_val),
        (X_test, ang_test, ids_test),
    ) = _load_and_process_data(load_cached_data)

    # Handle Debug Mode
    if Config.DEBUG:
        limit = Config.DEBUG_SIZE
        print(f"DEBUG MODE: Truncating datasets to {limit} samples.")
        X_train, ang_train, y_train = (
            X_train[:limit],
            ang_train[:limit],
            y_train[:limit],
        )
        X_val, ang_val, y_val = X_val[:limit], ang_val[:limit], y_val[:limit]
        X_test, ang_test, ids_test = X_test[:limit], ang_test[:limit], ids_test[:limit]

    # Define Transforms
    # We use RandomFlips for training.
    # Note: Input to transforms is (3, 75, 75) FloatTensor.
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No transforms for validation/test
    val_transform = None

    # Instantiate Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=val_transform)
    test_dataset = IcebergDataset(X_test, ang_test, y=None, transform=val_transform)

    # Instantiate Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, ids_test
