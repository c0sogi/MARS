import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import set_seed


def get_transforms(phase):
    """
    Returns torchvision transforms for the given phase.
    """
    if phase == "train":
        return transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )
    return None


class IcebergDataset(Dataset):
    def __init__(self, images, angles, targets=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            targets (np.ndarray, optional): Shape (N,)
            transform (callable, optional): Transform to apply to images
        """
        self.images = images
        self.angles = angles
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image: (3, 75, 75)
        img = self.images[idx]
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, target
        else:
            return img_tensor, angle_tensor


def load_and_process_json(json_path, cache_prefix, load_cached=True):
    """
    Loads JSON data, processes bands into 3-channel images, parses angles,
    and caches the result as .npy files.
    """
    # Define cache paths
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f"{cache_prefix}_X.npy")
    cache_ang = os.path.join(CACHE_DIR, f"{cache_prefix}_ang.npy")
    cache_y = os.path.join(CACHE_DIR, f"{cache_prefix}_y.npy")
    cache_ids = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Check cache
    files_exist = (
        os.path.exists(cache_X)
        and os.path.exists(cache_ang)
        and os.path.exists(cache_ids)
    )
    # y is optional (not in test)
    y_exists_if_needed = True
    if "train" in cache_prefix and not os.path.exists(cache_y):
        y_exists_if_needed = False

    if load_cached and files_exist and y_exists_if_needed:
        print(f"Loading cached data for {cache_prefix}...")
        X = np.load(cache_X)
        ang = np.load(cache_ang)
        ids = np.load(cache_ids)
        y = np.load(cache_y) if os.path.exists(cache_y) else None
        return X, ang, y, ids

    print(f"Processing raw data for {cache_prefix} from {json_path}...")
    # Read JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1: HH
    b1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
    )
    # Band 2: HV
    b2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
    )
    # Band 3: Average ((HH+HV)/2)
    b3 = (b1 + b2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1)

    # Process Incidence Angles
    # Convert 'na' to NaN
    df["inc_angle"] = pd.to_numeric(df["inc_angle"], errors="coerce")
    ang = df["inc_angle"].values.astype(np.float32)

    # Process IDs
    ids = df["id"].values

    # Process Targets
    y = None
    if "is_iceberg" in df.columns:
        y = df["is_iceberg"].values.astype(np.float32)

    # Save to cache
    print(f"Saving processed data to {CACHE_DIR}...")
    np.save(cache_X, X)
    np.save(cache_ang, ang)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, ang, y, ids


def get_dataloaders(debug_sample_size=None, load_cached=True):
    """
    Prepares and returns DataLoaders for train, val, and test sets.
    Performs median imputation for incidence angles.
    """
    set_seed(SEED)

    # 1. Load Metadata
    train_meta = pd.read_csv(TRAIN_META_PATH)
    val_meta = pd.read_csv(VAL_META_PATH)
    test_meta = pd.read_csv(TEST_META_PATH)

    # 2. Load Processed Arrays (Train contains both train and val splits)
    X_full, ang_full, y_full, _ = load_and_process_json(
        TRAIN_JSON_PATH, "train_full", load_cached
    )

    X_test, ang_test, _, _ = load_and_process_json(TEST_JSON_PATH, "test", load_cached)

    # 3. Split Training Data based on Metadata Indices
    train_indices = train_meta["original_index"].values
    val_indices = val_meta["original_index"].values
    test_indices = test_meta["original_index"].values

    X_train = X_full[train_indices]
    ang_train = ang_full[train_indices]
    y_train = y_full[train_indices]

    X_val = X_full[val_indices]
    ang_val = ang_full[val_indices]
    y_val = y_full[val_indices]

    # Ensure test alignment
    X_test = X_test[test_indices]
    ang_test = ang_test[test_indices]

    # 4. Impute Missing Incidence Angles
    # Calculate median from TRAIN set only (ignoring NaNs)
    valid_train_angles = ang_train[~np.isnan(ang_train)]
    if len(valid_train_angles) > 0:
        median_angle = np.median(valid_train_angles)
    else:
        median_angle = 0.0  # Fallback

    # Fill NaNs
    def impute(arr, val):
        mask = np.isnan(arr)
        arr[mask] = val
        return arr

    ang_train = impute(ang_train, median_angle)
    ang_val = impute(ang_val, median_angle)
    ang_test = impute(ang_test, median_angle)

    # 5. Debug Subsetting
    if debug_sample_size is not None:
        print(f"Debug mode: limiting dataset to {debug_sample_size} samples.")
        X_train = X_train[:debug_sample_size]
        ang_train = ang_train[:debug_sample_size]
        y_train = y_train[:debug_sample_size]

        X_val = X_val[:debug_sample_size]
        ang_val = ang_val[:debug_sample_size]
        y_val = y_val[:debug_sample_size]

        X_test = X_test[:debug_sample_size]
        ang_test = ang_test[:debug_sample_size]

    # 6. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, transform=get_transforms("train")
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=get_transforms("val"))
    test_dataset = IcebergDataset(
        X_test, ang_test, None, transform=get_transforms("test")
    )

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
