import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data.py")


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
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
        # Convert to tensor. X is already (C, H, W) from processing
        # Ensure float32 for PyTorch
        img = torch.from_numpy(self.X[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            return img, angle


def _process_json_data(json_path, metadata_df):
    """
    Helper to read JSON and extract specific samples based on metadata indices.
    Returns X (images), angles (raw), ids.
    """
    logger.info(f"Loading raw data from {json_path}...")
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Map original_index to the actual data object for O(1) access if indices are sorted,
    # but since raw_data is a list, we can just access by index.
    # Ensure metadata is sorted by original_index to optimize access if needed,
    # but random access in list is O(1).

    indices = metadata_df["original_index"].values
    ids = metadata_df["id"].values

    # Pre-allocate arrays
    num_samples = len(indices)
    img_size = Config.IMAGE_SIZE

    # Shape: (N, 3, 75, 75)
    X = np.zeros((num_samples, 3, img_size, img_size), dtype=np.float32)
    # Angles extracted from JSON (strings/floats) to be processed later
    raw_angles = []

    for i, original_idx in enumerate(indices):
        item = raw_data[original_idx]

        # Verify alignment (optional but good for safety)
        if item["id"] != ids[i]:
            raise ValueError(
                f"ID mismatch at index {i}: Meta {ids[i]} vs JSON {item['id']}"
            )

        # Band 1 (HH)
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(img_size, img_size)
        # Band 2 (HV)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(img_size, img_size)
        # Band 3 (Avg)
        b3 = (b1 + b2) / 2.0

        # Stack channels: (3, 75, 75)
        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

        raw_angles.append(item["inc_angle"])

    return X, raw_angles


def _load_and_process_data(load_cached_data=True):
    """
    Handles caching logic, data loading, reshaping, and imputation.
    """
    # Define cache paths
    paths = {
        "train": (Config.CACHE_TRAIN_X, Config.CACHE_TRAIN_Y, Config.CACHE_TRAIN_META),
        "val": (Config.CACHE_VAL_X, Config.CACHE_VAL_Y, Config.CACHE_VAL_META),
        "test": (Config.CACHE_TEST_X, None, Config.CACHE_TEST_META),
    }

    # Check if all cache files exist
    all_cached = True
    if load_cached_data:
        for mode, (px, py, pmeta) in paths.items():
            if not os.path.exists(px) or not os.path.exists(pmeta):
                all_cached = False
                break
            if py and not os.path.exists(py):
                all_cached = False
                break
    else:
        all_cached = False

    if all_cached:
        logger.info("Loading data from cache...")
        data = {}
        for mode, (px, py, pmeta) in paths.items():
            data[f"X_{mode}"] = np.load(px)
            data[f"meta_{mode}"] = np.load(pmeta)  # Stores angles
            if py:
                data[f"y_{mode}"] = np.load(py)
        return data

    logger.info("Cache missing or reload requested. Processing raw data...")

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # 2. Impute Angles
    # Calculate median from training set where angle is not NaN
    # The metadata CSVs have NaN for 'na' values.
    train_angles = df_train["inc_angle"].dropna()
    median_angle = train_angles.median()
    logger.info(f"Median incidence angle calculated from train set: {median_angle}")

    def fill_angles(df):
        # Fill NaN with median
        return df["inc_angle"].fillna(median_angle).values.astype(np.float32)

    ang_train = fill_angles(df_train)
    ang_val = fill_angles(df_val)
    ang_test = fill_angles(df_test)

    # 3. Process Images
    # We need to load train.json for train/val sets, and test.json for test set

    # Process Train
    X_train, _ = _process_json_data(Config.TRAIN_JSON, df_train)
    y_train = df_train["is_iceberg"].values.astype(np.float32)

    # Process Val
    X_val, _ = _process_json_data(Config.TRAIN_JSON, df_val)
    y_val = df_val["is_iceberg"].values.astype(np.float32)

    # Process Test
    X_test, _ = _process_json_data(Config.TEST_JSON, df_test)

    # 4. Save to Cache
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    np.save(Config.CACHE_TRAIN_X, X_train)
    np.save(Config.CACHE_TRAIN_Y, y_train)
    np.save(Config.CACHE_TRAIN_META, ang_train)

    np.save(Config.CACHE_VAL_X, X_val)
    np.save(Config.CACHE_VAL_Y, y_val)
    np.save(Config.CACHE_VAL_META, ang_val)

    np.save(Config.CACHE_TEST_X, X_test)
    np.save(Config.CACHE_TEST_META, ang_test)

    logger.info("Data processed and cached.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": ang_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": ang_val,
        "X_test": X_test,
        "meta_test": ang_test,
    }


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data = _load_and_process_data(load_cached_data=load_cached_data)

    # Define Augmentations for Training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Datasets
    train_dataset = IcebergDataset(
        X=data["X_train"],
        angles=data["meta_train"],
        y=data["y_train"],
        transform=train_transform,
    )

    val_dataset = IcebergDataset(
        X=data["X_val"],
        angles=data["meta_val"],
        y=data["y_val"],
        transform=None,  # No TTA/Augmentation for Val
    )

    test_dataset = IcebergDataset(
        X=data["X_test"],
        angles=data["meta_test"],
        y=None,
        transform=None,  # No TTA for Test
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
