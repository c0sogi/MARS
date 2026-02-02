import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import (
    WORKING_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_JSON,
    TEST_JSON,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_HEIGHT,
    IMG_WIDTH,
    SEED,
)
from library.utils import set_seed


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, H, W, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): IDs of shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # Shape: (75, 75, 3), dtype: float32
        angle = self.angles[idx]  # Scalar float

        # Apply transforms
        # ToTensor() converts (H, W, C) -> (C, H, W).
        # Since data is float, it does not scale [0-255] to [0-1], just permutes.
        if self.transform:
            img = self.transform(img)
        else:
            # Fallback if no transform is provided
            img = torch.from_numpy(img).float().permute(2, 0, 1)

        # Convert angle to tensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Get ID
        id_val = self.ids[idx] if self.ids is not None else str(idx)

        # Return tuple based on availability of label
        if self.y is not None:
            label_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle_tensor, label_tensor, id_val
        else:
            return img, angle_tensor, id_val


def _process_split(meta_path, json_path, split_name, load_cached_data=True):
    """
    Loads data for a specific split, processing from raw JSON if not cached.

    Args:
        meta_path (str): Path to the metadata CSV.
        json_path (str): Path to the raw JSON file.
        split_name (str): Name of the split (train, val, test) for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        X (np.ndarray): Images (N, 75, 75, 3)
        angles (np.ndarray): Incidence angles (N,)
        y (np.ndarray or None): Labels (N,)
        ids (np.ndarray): IDs (N,)
    """
    # Define cache paths
    cache_X = os.path.join(WORKING_DIR, f"X_{split_name}.npy")
    cache_angles = os.path.join(WORKING_DIR, f"angles_{split_name}.npy")
    cache_y = os.path.join(WORKING_DIR, f"y_{split_name}.npy")
    cache_ids = os.path.join(WORKING_DIR, f"ids_{split_name}.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_X)
        and os.path.exists(cache_angles)
        and os.path.exists(cache_ids)
    )
    # y is optional for test set
    if split_name != "test":
        cache_exists = cache_exists and os.path.exists(cache_y)

    # 1. Load from cache if requested and available
    if load_cached_data and cache_exists:
        print(f"Loading {split_name} data from cache...")
        X = np.load(cache_X)
        angles = np.load(cache_angles)
        ids = np.load(cache_ids)
        y = np.load(cache_y) if split_name != "test" else None
        return X, angles, y, ids

    # 2. Process from scratch
    print(f"Processing {split_name} data from raw files...")

    # Load metadata
    df_meta = pd.read_csv(meta_path)
    target_ids = set(df_meta["id"].values)

    # Load raw JSON
    # Note: Loading entire JSON into memory is feasible given the constraints (220GB RAM)
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Filter and sort raw data to match metadata order
    # Create a dict for O(1) lookup
    data_map = {item["id"]: item for item in raw_data if item["id"] in target_ids}

    # Lists to store processed data
    X_list = []
    angles_list = []
    y_list = []
    ids_list = []

    # Iterate through metadata to ensure correct order and stratification
    for _, row in df_meta.iterrows():
        img_id = row["id"]
        item = data_map[img_id]

        # Process Bands
        # Flattened 75x75 = 5625
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)

        # Construct 3rd channel: Average of HH and HV
        b3 = (b1 + b2) / 2.0

        # Stack to (75, 75, 3)
        img = np.dstack((b1, b2, b3))
        X_list.append(img)

        # Process Angle
        # Use metadata inc_angle which handles 'na' coercion
        angles_list.append(row["inc_angle"])

        # Process ID
        ids_list.append(img_id)

        # Process Target (if available)
        if "is_iceberg" in row:
            y_list.append(row["is_iceberg"])

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    angles = np.array(angles_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    # Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    np.save(cache_X, X)
    np.save(cache_angles, angles)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, angles, y, ids


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(SEED)

    # 1. Load Data Splits
    X_train, angles_train, y_train, ids_train = _process_split(
        TRAIN_META_PATH, TRAIN_JSON, "train", load_cached_data
    )
    X_val, angles_val, y_val, ids_val = _process_split(
        VAL_META_PATH, TRAIN_JSON, "val", load_cached_data
    )
    X_test, angles_test, _, ids_test = _process_split(
        TEST_META_PATH, TEST_JSON, "test", load_cached_data
    )

    # 2. Impute Missing Incidence Angles
    # Calculate median from training set (ignoring NaNs)
    angle_median = np.nanmedian(angles_train)

    # Fill NaNs
    angles_train = np.nan_to_num(angles_train, nan=angle_median)
    angles_val = np.nan_to_num(angles_val, nan=angle_median)
    angles_test = np.nan_to_num(angles_test, nan=angle_median)

    # 3. Debug Subsetting
    if debug:
        limit = 100
        X_train, angles_train, y_train, ids_train = (
            X_train[:limit],
            angles_train[:limit],
            y_train[:limit],
            ids_train[:limit],
        )
        X_val, angles_val, y_val, ids_val = (
            X_val[:limit],
            angles_val[:limit],
            y_val[:limit],
            ids_val[:limit],
        )
        X_test, angles_test, ids_test = (
            X_test[:limit],
            angles_test[:limit],
            ids_test[:limit],
        )

    # 4. Define Transforms
    # Train: Augmentation + ToTensor
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (H,W,C) -> (C,H,W), no scaling for float
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # Val/Test: ToTensor only
    eval_transform = transforms.Compose([transforms.ToTensor()])

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val, angles_val, y_val, ids_val, transform=eval_transform
    )
    test_dataset = IcebergDataset(
        X_test, angles_test, None, ids_test, transform=eval_transform
    )

    # 6. Create DataLoaders
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
