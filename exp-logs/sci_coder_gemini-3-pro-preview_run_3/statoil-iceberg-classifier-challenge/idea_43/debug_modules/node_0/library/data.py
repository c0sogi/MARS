import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.utils import set_seed

CACHE_DIR = "./working/idea_43/"


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply augmentations
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return (image, angle, label) for train/val, (image, angle, id) for test
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            img_id = self.ids[idx]
            return img_tensor, angle_tensor, img_id


def process_data(load_cached_data=True):
    """
    Loads data from cache or processes it from raw JSON/CSV files.
    Implements caching mechanism using .npy files.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "angles_train": os.path.join(CACHE_DIR, "angles_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "angles_test": os.path.join(CACHE_DIR, "angles_test.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
    }

    # 1. Try Loading Cache
    if load_cached_data:
        if all(os.path.exists(f) for f in cache_files.values()):
            X_train = np.load(cache_files["X_train"])
            angles_train = np.load(cache_files["angles_train"])
            y_train = np.load(cache_files["y_train"])
            X_test = np.load(cache_files["X_test"])
            angles_test = np.load(cache_files["angles_test"])
            ids_test = np.load(cache_files["ids_test"])
            return (X_train, angles_train, y_train), (X_test, angles_test, ids_test)

    # 2. Process from Scratch

    # Load Metadata
    train_meta = pd.read_csv("./metadata/train.csv")
    val_meta = pd.read_csv("./metadata/val.csv")
    test_meta = pd.read_csv("./metadata/test.csv")

    # Combine train and val metadata to form the full training set for CV
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Load Raw JSONs
    with open("./input/train.json", "r") as f:
        raw_train = json.load(f)
    with open("./input/test.json", "r") as f:
        raw_test = json.load(f)

    # Create lookup dicts for fast access
    train_dict = {item["id"]: item for item in raw_train}
    test_dict = {item["id"]: item for item in raw_test}

    def extract_features(meta_df, data_dict):
        X_list = []
        angles_list = []
        ids_list = []
        y_list = []

        for _, row in meta_df.iterrows():
            eid = row["id"]
            item = data_dict[eid]

            # Reshape bands (75x75)
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)

            # Generate 3rd channel (Average)
            b3 = (b1 + b2) / 2.0

            # Stack channels: (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            X_list.append(img)

            # Incidence Angle (use metadata value which handles 'na' -> NaN)
            angles_list.append(row["inc_angle"])
            ids_list.append(eid)

            if "is_iceberg" in row:
                y_list.append(row["is_iceberg"])

        return (
            np.array(X_list, dtype=np.float32),
            np.array(angles_list, dtype=np.float32),
            np.array(ids_list),
            np.array(y_list, dtype=np.float32) if y_list else None,
        )

    # Extract data
    X_train, angles_train, _, y_train = extract_features(full_train_meta, train_dict)
    X_test, angles_test, ids_test, _ = extract_features(test_meta, test_dict)

    # 3. Median Imputation for Incidence Angles
    # Calculate median from valid training angles only
    valid_angles = angles_train[~np.isnan(angles_train)]
    median_angle = np.median(valid_angles)

    # Fill NaNs
    angles_train[np.isnan(angles_train)] = median_angle
    angles_test[np.isnan(angles_test)] = median_angle

    # 4. Save to Cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return (X_train, angles_train, y_train), (X_test, angles_test, ids_test)


def get_loaders(fold=0, n_folds=5, batch_size=32, num_workers=2, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold using Stratified K-Fold.
    """
    # Ensure reproducibility
    set_seed(42)

    # Get processed data
    (X_train_full, angles_train_full, y_train_full), (X_test, angles_test, ids_test) = (
        process_data(load_cached_data)
    )

    # Stratified Split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Generate splits (y needs to be provided for stratification)
    # We convert y to int for stratification logic, though sklearn handles floats if they are discrete classes
    splits = list(skf.split(np.zeros(len(y_train_full)), y_train_full.astype(int)))

    if fold < 0 or fold >= n_folds:
        raise ValueError(f"Fold {fold} is out of range for {n_folds} folds.")

    train_idx, val_idx = splits[fold]

    # Subset data
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    ang_train, ang_val = angles_train_full[train_idx], angles_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    # Define Transforms (Training only)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    train_ds = IcebergDataset(X_train, ang_train, y_train, transform=train_transform)
    val_ds = IcebergDataset(X_val, ang_val, y_val, transform=None)
    test_ds = IcebergDataset(X_test, angles_test, ids=ids_test, transform=None)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
