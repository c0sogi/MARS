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
    PyTorch Dataset for the Iceberg/Ship classification task.
    Handles 3-channel image tensors, incidence angles, and labels/IDs.
    """

    def __init__(self, X, angles, labels=None, ids=None, transform=None, mode="train"):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            labels (np.ndarray, optional): Target labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Transformations to apply to the images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.X = X
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        x = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]  # Scalar

        # Convert to tensors
        x_tensor = torch.from_numpy(x).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float)

        # Apply augmentations (only affects image tensor)
        if self.transform:
            x_tensor = self.transform(x_tensor)

        # Return appropriate tuple based on mode
        if self.mode == "test":
            # For test, return ID instead of label
            id_val = self.ids[idx]
            return x_tensor, angle_tensor, id_val
        else:
            # For train/val, return label
            y_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
            return x_tensor, angle_tensor, y_tensor


def get_median_angle():
    """
    Calculates the median incidence angle from the training metadata.
    Used to impute missing 'na' values in the dataset.
    """
    df_train = pd.read_csv(Config.TRAIN_META_CSV)
    # Coerce errors to NaN to handle 'na' strings, then compute median
    angles = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    return angles.median()


def process_data(mode, median_angle, load_cached_data=True):
    """
    Loads and processes data for a specific split (train, val, test).
    Uses caching to speed up subsequent runs.

    Args:
        mode (str): 'train', 'val', or 'test'.
        median_angle (float): Value to use for imputing missing angles.
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        tuple: (X, angles, y, ids) where y or ids might be None depending on mode.
    """
    # Define cache file paths
    # We use specific paths from Config if available, or generate standard ones
    if mode == "train":
        cache_X = Config.CACHE_X_TRAIN
        cache_angle = Config.CACHE_ANGLE_TRAIN
        cache_y = Config.CACHE_Y_TRAIN
        cache_ids = Config.CACHE_IDS_TRAIN
        meta_path = Config.TRAIN_META_CSV
    elif mode == "test":
        cache_X = Config.CACHE_X_TEST
        cache_angle = Config.CACHE_ANGLE_TEST
        cache_y = os.path.join(Config.CACHE_DIR, "y_test.npy")  # Not used usually
        cache_ids = Config.CACHE_IDS_TEST
        meta_path = Config.TEST_META_CSV
    else:  # val
        cache_X = os.path.join(Config.CACHE_DIR, "X_val.npy")
        cache_angle = os.path.join(Config.CACHE_DIR, "angle_val.npy")
        cache_y = os.path.join(Config.CACHE_DIR, "y_val.npy")
        cache_ids = os.path.join(Config.CACHE_DIR, "ids_val.npy")
        meta_path = Config.VAL_META_CSV

    # Check if all required cache files exist
    files_exist = os.path.exists(cache_X) and os.path.exists(cache_angle)
    if mode in ["train", "val"]:
        files_exist = files_exist and os.path.exists(cache_y)
    else:
        files_exist = files_exist and os.path.exists(cache_ids)

    # Attempt to load from cache
    if load_cached_data and files_exist:
        try:
            X = np.load(cache_X)
            angles = np.load(cache_angle)
            if mode in ["train", "val"]:
                y = np.load(cache_y)
                return X, angles, y, None
            else:
                ids = np.load(cache_ids)
                return X, angles, None, ids
        except Exception:
            # If loading fails, fall back to processing
            pass

    # --- Process from Raw Data ---
    print(f"Processing {mode} data from raw JSON...")

    # Load Metadata
    df_meta = pd.read_csv(meta_path)

    # Load Raw JSON
    # Identify which file to load based on the first row of metadata
    source_file = df_meta["source_file"].iloc[0]
    json_path = os.path.join(Config.INPUT_DIR, source_file)

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Initialize arrays
    num_samples = len(df_meta)
    img_size = Config.IMAGE_SIZE

    X = np.zeros((num_samples, 3, img_size, img_size), dtype=np.float32)
    angles = np.zeros(num_samples, dtype=np.float32)

    if mode in ["train", "val"]:
        y = np.zeros(num_samples, dtype=np.float32)
        ids = None
    else:
        y = None
        ids = np.empty(num_samples, dtype=object)

    # Process each sample
    # Note: raw_data is a list of dicts. Metadata has 'original_index' to map directly.
    for i, row in df_meta.iterrows():
        orig_idx = int(row["original_index"])
        item = raw_data[orig_idx]

        # Sanity check ID
        if item["id"] != row["id"]:
            # Fallback search if index is mismatched
            item = next((x for x in raw_data if x["id"] == row["id"]), None)
            if item is None:
                raise ValueError(f"ID {row['id']} not found in {source_file}")

        # Construct 3-channel image
        # Band 1: HH
        b1 = np.array(item["band_1"]).reshape(img_size, img_size)
        # Band 2: HV
        b2 = np.array(item["band_2"]).reshape(img_size, img_size)
        # Band 3: Average
        b3 = (b1 + b2) / 2.0

        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

        # Process Angle (Impute if necessary)
        # Metadata inc_angle is already numeric with NaNs where appropriate
        ang = row["inc_angle"]
        if pd.isna(ang):
            ang = median_angle
        angles[i] = ang

        # Process Label or ID
        if mode in ["train", "val"]:
            y[i] = row["is_iceberg"]
        else:
            ids[i] = row["id"]

    # Save to cache
    np.save(cache_X, X)
    np.save(cache_angle, angles)
    if mode in ["train", "val"]:
        np.save(cache_y, y)
    else:
        np.save(cache_ids, ids)

    return X, angles, y, ids


def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Compute global imputation value for angles
    median_angle = get_median_angle()

    # 2. Load Data
    X_train, ang_train, y_train, _ = process_data(
        "train", median_angle, load_cached_data
    )
    X_val, ang_val, y_val, _ = process_data("val", median_angle, load_cached_data)
    X_test, ang_test, _, ids_test = process_data("test", median_angle, load_cached_data)

    # 3. Define Augmentations
    # We use geometric augmentations suitable for radar data
    train_transform = None
    if Config.AUGMENT_HFLIP or Config.AUGMENT_VFLIP:
        transforms_list = []
        if Config.AUGMENT_HFLIP:
            transforms_list.append(transforms.RandomHorizontalFlip())
        if Config.AUGMENT_VFLIP:
            transforms_list.append(transforms.RandomVerticalFlip())
        train_transform = transforms.Compose(transforms_list)

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, labels=y_train, transform=train_transform, mode="train"
    )
    val_dataset = IcebergDataset(
        X_val, ang_val, labels=y_val, transform=None, mode="val"
    )
    test_dataset = IcebergDataset(
        X_test, ang_test, ids=ids_test, transform=None, mode="test"
    )

    # 5. Create DataLoaders
    # Pin memory enables faster data transfer to CUDA devices
    use_pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
