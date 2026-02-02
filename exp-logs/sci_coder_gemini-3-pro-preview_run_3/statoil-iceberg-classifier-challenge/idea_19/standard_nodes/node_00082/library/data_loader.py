import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.utils import impute_inc_angle


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        """
        PyTorch Dataset for Iceberg/Ship classification.

        Args:
            X (np.ndarray): Input images of shape (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Target labels of shape (N,).
            transform (callable, optional): Torchvision transforms to apply.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load image from numpy array
        # Input shape is (75, 75, 3) -> (H, W, C)
        img = self.X[idx]

        # Convert to Tensor and rearrange to (C, H, W)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)

        # Apply augmentations if provided
        if self.transform:
            img = self.transform(img)

        # Get incidence angle
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Return with label if available
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            return img, angle


def get_transforms(mode="train"):
    """
    Returns the data augmentation pipeline.

    Args:
        mode (str): 'train' or 'test'.

    Returns:
        torchvision.transforms.Compose or None
    """
    if mode == "train":
        return transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )
    else:
        return None


def _process_json_data(data_list, indices):
    """
    Internal helper to extract bands from raw JSON list and stack them.

    Args:
        data_list (list): List of dictionaries loaded from JSON.
        indices (list/array): List of indices to extract.

    Returns:
        np.ndarray: Array of shape (N, 75, 75, 3).
    """
    X = []
    for idx in indices:
        item = data_list[idx]

        # Reshape flattened bands to 75x75
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)

        # Synthetic 3rd band: Average of HH and HV
        b3 = (b1 + b2) / 2.0

        # Stack to create 3-channel image (H, W, C)
        img = np.dstack((b1, b2, b3))
        X.append(img)

    return np.array(X, dtype=np.float32)


def load_data(load_cached_data=True):
    """
    Loads dataset, handling caching, metadata splitting, and imputation.

    Args:
        load_cached_data (bool): If True, tries to load processed .npy files from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays:
              'X_train', 'angle_train', 'y_train',
              'X_val', 'angle_val', 'y_val',
              'X_test', 'angle_test', 'ids_test'
    """
    cache_dir = "./working/idea_19"
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "X_train": "X_train.npy",
        "angle_train": "angle_train.npy",
        "y_train": "y_train.npy",
        "X_val": "X_val.npy",
        "angle_val": "angle_val.npy",
        "y_val": "y_val.npy",
        "X_test": "X_test.npy",
        "angle_test": "angle_test.npy",
        "ids_test": "ids_test.npy",
    }

    # Check if all cache files exist
    cache_exists = all(
        os.path.exists(os.path.join(cache_dir, f)) for f in cache_files.values()
    )

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        data = {}
        for key, filename in cache_files.items():
            path = os.path.join(cache_dir, filename)
            data[key] = np.load(path, allow_pickle=True)
            # Ensure IDs are strings (numpy might load as object)
            if key == "ids_test":
                data[key] = data[key].astype(str)
        return data

    print("Processing data from scratch...")

    # 1. Load Metadata
    train_meta = pd.read_csv("./metadata/train.csv")
    val_meta = pd.read_csv("./metadata/val.csv")
    test_meta = pd.read_csv("./metadata/test.csv")

    # 2. Impute Incidence Angles
    # Calculate median from training set ONLY to prevent leakage
    train_meta, median = impute_inc_angle(train_meta)
    # Apply training median to validation and test sets
    val_meta, _ = impute_inc_angle(val_meta, train_median=median)
    test_meta, _ = impute_inc_angle(test_meta, train_median=median)

    # 3. Load Raw JSON
    # 'train.json' contains the source for both our train and val splits
    with open("./input/train.json", "r") as f:
        raw_train_source = json.load(f)

    with open("./input/test.json", "r") as f:
        raw_test_source = json.load(f)

    # 4. Process Splits
    print("Processing Train split...")
    X_train = _process_json_data(raw_train_source, train_meta["original_index"].values)
    angle_train = train_meta["inc_angle"].values.astype(np.float32)
    y_train = train_meta["is_iceberg"].values.astype(np.float32)

    print("Processing Val split...")
    X_val = _process_json_data(raw_train_source, val_meta["original_index"].values)
    angle_val = val_meta["inc_angle"].values.astype(np.float32)
    y_val = val_meta["is_iceberg"].values.astype(np.float32)

    print("Processing Test split...")
    X_test = _process_json_data(raw_test_source, test_meta["original_index"].values)
    angle_test = test_meta["inc_angle"].values.astype(np.float32)
    ids_test = test_meta["id"].values.astype(str)

    # 5. Save to Cache
    print("Saving processed data to cache...")
    np.save(os.path.join(cache_dir, cache_files["X_train"]), X_train)
    np.save(os.path.join(cache_dir, cache_files["angle_train"]), angle_train)
    np.save(os.path.join(cache_dir, cache_files["y_train"]), y_train)

    np.save(os.path.join(cache_dir, cache_files["X_val"]), X_val)
    np.save(os.path.join(cache_dir, cache_files["angle_val"]), angle_val)
    np.save(os.path.join(cache_dir, cache_files["y_val"]), y_val)

    np.save(os.path.join(cache_dir, cache_files["X_test"]), X_test)
    np.save(os.path.join(cache_dir, cache_files["angle_test"]), angle_test)
    np.save(os.path.join(cache_dir, cache_files["ids_test"]), ids_test)

    return {
        "X_train": X_train,
        "angle_train": angle_train,
        "y_train": y_train,
        "X_val": X_val,
        "angle_val": angle_val,
        "y_val": y_val,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }
