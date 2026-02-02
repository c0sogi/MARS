import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from library.utils import set_seed

# ==========================================
# Configuration & Constants
# ==========================================
CACHE_DIR = "./working/idea_46/"
INPUT_DIR = "./input/"
METADATA_DIR = "./metadata/"


# ==========================================
# Dataset Class
# ==========================================
class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75)
            angles (np.ndarray): Incidence angles of shape (N,)
            y (np.ndarray, optional): Labels of shape (N,). Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # (3, 75, 75)
        angle = self.angles[idx]  # scalar

        # Convert to tensors
        img_tensor = torch.from_numpy(img)
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        # Apply transforms (Augmentation)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return with or without label
        if self.y is not None:
            label = self.y[idx]
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


# ==========================================
# Data Processing & Loading
# ==========================================
def load_data(load_cached_data=True):
    """
    Loads processed data from cache or processes it from scratch.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing numpy arrays for train/val/test splits.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "X_train": "X_train.npy",
        "angle_train": "angle_train.npy",
        "y_train": "y_train.npy",
        "ids_train": "ids_train.npy",
        "X_val": "X_val.npy",
        "angle_val": "angle_val.npy",
        "y_val": "y_val.npy",
        "ids_val": "ids_val.npy",
        "X_test": "X_test.npy",
        "angle_test": "angle_test.npy",
        "ids_test": "ids_test.npy",
    }

    # Check if all cache files exist
    all_cached = all(
        os.path.exists(os.path.join(CACHE_DIR, f)) for f in cache_files.values()
    )

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        data = {}
        for k, v in cache_files.items():
            path = os.path.join(CACHE_DIR, v)
            if "ids" in k:
                data[k] = np.load(path, allow_pickle=True)
            else:
                data[k] = np.load(path)
        return data

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Load Raw JSON Data into memory
    # We load both files once to map them efficiently
    raw_data_map = {}
    with open(os.path.join(INPUT_DIR, "train.json"), "r") as f:
        raw_data_map["train.json"] = json.load(f)
    with open(os.path.join(INPUT_DIR, "test.json"), "r") as f:
        raw_data_map["test.json"] = json.load(f)

    def process_split(df, is_test=False):
        X_list, ang_list, id_list, y_list = [], [], [], []

        for _, row in df.iterrows():
            source_file = row["source_file"]
            original_idx = int(row["original_index"])

            # Retrieve raw item
            item = raw_data_map[source_file][original_idx]

            # Process ID
            id_list.append(item["id"])

            # Process Images
            # Flattened list -> 75x75
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
            # Synthetic 3rd band: Average
            b3 = (b1 + b2) / 2.0

            # Stack to (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            X_list.append(img)

            # Process Angle
            # Use metadata value (which has 'na' converted to NaN or float)
            ang_list.append(row["inc_angle"])

            # Process Label
            if not is_test:
                y_list.append(item["is_iceberg"])

        # Convert lists to numpy arrays
        X_arr = np.array(X_list, dtype=np.float32)
        ang_arr = np.array(ang_list, dtype=np.float32)
        id_arr = np.array(id_list)

        if is_test:
            return X_arr, ang_arr, id_arr, None
        else:
            y_arr = np.array(y_list, dtype=np.float32)
            return X_arr, ang_arr, id_arr, y_arr

    # Process each split
    X_train, ang_train, ids_train, y_train = process_split(train_meta, is_test=False)
    X_val, ang_val, ids_val, y_val = process_split(val_meta, is_test=False)
    X_test, ang_test, ids_test, _ = process_split(test_meta, is_test=True)

    # Impute Missing Angles
    # Calculate median from valid training angles only
    valid_train_angles = ang_train[~np.isnan(ang_train)]
    median_angle = np.median(valid_train_angles) if len(valid_train_angles) > 0 else 0.0

    # Fill NaNs in all sets using training median
    ang_train = np.nan_to_num(ang_train, nan=median_angle)
    ang_val = np.nan_to_num(ang_val, nan=median_angle)
    ang_test = np.nan_to_num(ang_test, nan=median_angle)

    # Pack into dictionary
    data = {
        "X_train": X_train,
        "angle_train": ang_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_val": X_val,
        "angle_val": ang_val,
        "y_val": y_val,
        "ids_val": ids_val,
        "X_test": X_test,
        "angle_test": ang_test,
        "ids_test": ids_test,
    }

    # Save to cache
    for k, v in data.items():
        np.save(os.path.join(CACHE_DIR, cache_files[k]), v)

    return data


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Generates PyTorch DataLoaders for Train, Validation, and Test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Load data
    data = load_data(load_cached_data)

    # Define Augmentations for Training
    train_transform = T.Compose(
        [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        data["X_train"], data["angle_train"], data["y_train"], transform=train_transform
    )
    val_dataset = IcebergDataset(
        data["X_val"], data["angle_val"], data["y_val"], transform=None
    )
    test_dataset = IcebergDataset(
        data["X_test"], data["angle_test"], y=None, transform=None
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, data["ids_test"]
