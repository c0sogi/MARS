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
    PyTorch Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, 75, 75, 3).
            angles (np.ndarray): Array of shape (N,).
            labels (np.ndarray, optional): Array of shape (N,).
            ids (np.ndarray, optional): Array of strings of shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
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
        image = self.images[idx]  # (75, 75, 3)
        angle = self.angles[idx]  # scalar

        # Apply transforms
        # Note: transforms.ToTensor() converts (H, W, C) -> (C, H, W).
        # For float32 numpy arrays, it does NOT scale to [0, 1], which preserves dB values.
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float()

        # Convert angle to tensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Prepare return dictionary
        sample = {"image": image_tensor, "angle": angle_tensor}

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return sample, label_tensor

        return sample


def get_train_median_angle():
    """
    Calculates the median incidence angle from the training metadata.
    Used for imputing missing values across all splits.
    """
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    median_angle = df_train["inc_angle"].median()
    return median_angle


def process_and_cache_data(
    mode, metadata_path, json_path, median_angle, load_cached_data=True
):
    """
    Loads raw data, processes it (bands, angle imputation), and caches it.

    Args:
        mode (str): 'train', 'val', or 'test'.
        metadata_path (str): Path to the metadata CSV.
        json_path (str): Path to the raw JSON file.
        median_angle (float): Value to fill NaNs in incidence angle.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, angles, y, ids)
    """
    # Define cache paths
    cache_X = os.path.join(Config.CACHE_DIR, f"X_{mode}.npy")
    cache_angles = os.path.join(Config.CACHE_DIR, f"angle_{mode}.npy")
    cache_y = os.path.join(Config.CACHE_DIR, f"y_{mode}.npy")
    cache_ids = os.path.join(Config.CACHE_DIR, f"ids_{mode}.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(cache_X)
            and os.path.exists(cache_angles)
            and os.path.exists(cache_ids)
        ):
            # Check if y exists (only for train/val)
            if mode in ["train", "val"] and not os.path.exists(cache_y):
                pass  # Cache incomplete, proceed to process
            else:
                # print(f"Loading {mode} data from cache...")
                X = np.load(cache_X)
                angles = np.load(cache_angles)
                ids = np.load(cache_ids)
                y = np.load(cache_y) if mode in ["train", "val"] else None
                return X, angles, y, ids

    # 2. Process from Scratch
    # print(f"Processing {mode} data from raw files...")

    # Load Metadata
    df_meta = pd.read_csv(metadata_path)
    target_ids = set(df_meta["id"].values)

    # Load Raw JSON
    # We load the full JSON and filter.
    # pd.read_json is convenient but we need to ensure we match the metadata order.
    # To save memory/time, we can just iterate the list if we used json.load,
    # but pandas is robust for the band extraction.
    df_raw = pd.read_json(json_path)

    # Filter raw data to include only rows present in the current split (metadata)
    df_split = df_raw[df_raw["id"].isin(target_ids)].copy()

    # Merge with metadata to ensure order and get cleaned angles/labels if needed
    # Note: Metadata has the ground truth 'is_iceberg' and cleaned 'inc_angle' (with NaNs)
    # We drop 'inc_angle' from raw to use the one from metadata which might be pre-cleaned
    # (though here we do imputation manually).
    # Actually, the metadata file has 'inc_angle' with NaNs.

    # Align df_split with df_meta
    df_merged = pd.merge(
        df_meta, df_split[["id", "band_1", "band_2"]], on="id", how="left"
    )

    # Extract Bands
    # Stack band_1 and band_2
    b1 = np.array(df_merged["band_1"].tolist())
    b2 = np.array(df_merged["band_2"].tolist())

    # Reshape to (N, 75, 75)
    b1 = b1.reshape(-1, 75, 75)
    b2 = b2.reshape(-1, 75, 75)

    # Calculate Band 3 (Average)
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 75, 75, 3) -> (H, W, C) for consistency with standard image formats
    # We use stack on last axis
    X = np.stack([b1, b2, b3], axis=-1).astype(np.float32)

    # Extract and Impute Angles
    angles = df_merged["inc_angle"].values.astype(np.float32)
    # Impute NaNs with training median
    angles[np.isnan(angles)] = median_angle

    # Extract IDs
    ids = df_merged["id"].values

    # Extract Labels (if available)
    y = None
    if "is_iceberg" in df_merged.columns:
        y = df_merged["is_iceberg"].values.astype(np.float32)

    # 3. Save to Cache
    np.save(cache_X, X)
    np.save(cache_angles, angles)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, angles, y, ids


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Get Global Statistics for Imputation
    train_median_angle = get_train_median_angle()

    # 2. Process Data
    # Train
    X_train, ang_train, y_train, ids_train = process_and_cache_data(
        "train",
        Config.TRAIN_META_PATH,
        Config.TRAIN_JSON,
        train_median_angle,
        load_cached_data,
    )

    # Val
    X_val, ang_val, y_val, ids_val = process_and_cache_data(
        "val",
        Config.VAL_META_PATH,
        Config.TRAIN_JSON,
        train_median_angle,
        load_cached_data,
    )

    # Test
    X_test, ang_test, y_test, ids_test = process_and_cache_data(
        "test",
        Config.TEST_META_PATH,
        Config.TEST_JSON,
        train_median_angle,
        load_cached_data,
    )

    # Debugging: Subset data if enabled
    if Config.DEBUG:
        limit = Config.MAX_SAMPLES if Config.MAX_SAMPLES else 100
        X_train, ang_train, y_train, ids_train = (
            X_train[:limit],
            ang_train[:limit],
            y_train[:limit],
            ids_train[:limit],
        )
        X_val, ang_val, y_val, ids_val = (
            X_val[:limit],
            ang_val[:limit],
            y_val[:limit],
            ids_val[:limit],
        )
        X_test, ang_test, ids_test = X_test[:limit], ang_test[:limit], ids_test[:limit]

    # 3. Define Transforms
    # Train: ToTensor (HWC->CHW) + Augmentation
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # Val/Test: ToTensor only
    eval_transform = transforms.Compose([transforms.ToTensor()])

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val, ang_val, y_val, ids_val, transform=eval_transform
    )
    test_dataset = IcebergDataset(
        X_test, ang_test, None, ids_test, transform=eval_transform
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
