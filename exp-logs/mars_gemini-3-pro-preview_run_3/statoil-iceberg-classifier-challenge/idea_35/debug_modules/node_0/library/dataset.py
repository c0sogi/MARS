import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed

CACHE_DIR = "./working/idea_35/"


class IcebergDataset(Dataset):
    def __init__(self, X, angles, ids, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            ids (np.ndarray): Image IDs.
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.ids = ids
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]
        id_ = self.ids[idx]

        # Convert to tensors
        # Data is float32, keep as is.
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float)

        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor, id_


def get_transforms(phase):
    """
    Returns transforms for the given phase.
    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        return None


def process_split(metadata_path, json_path, split_name, load_cached_data):
    """
    Loads data for a specific split, using cache if available.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    path_X = os.path.join(CACHE_DIR, f"X_{split_name}.npy")
    path_angles = os.path.join(CACHE_DIR, f"angles_{split_name}.npy")
    path_y = os.path.join(CACHE_DIR, f"y_{split_name}.npy")
    path_ids = os.path.join(CACHE_DIR, f"ids_{split_name}.npy")

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(path_X)
            and os.path.exists(path_angles)
            and os.path.exists(path_ids)
        ):
            try:
                # Check if y exists (it won't for test)
                y_exists = os.path.exists(path_y)
                if split_name == "test" or y_exists:
                    print(f"Loading cached data for {split_name}...")
                    X = np.load(path_X)
                    angles = np.load(path_angles)
                    ids = np.load(path_ids)
                    y = np.load(path_y) if y_exists else None
                    return X, angles, y, ids
            except Exception as e:
                print(f"Error loading cache for {split_name}: {e}. Re-processing.")

    print(f"Processing data for {split_name} from scratch...")

    # Load Metadata
    df_meta = pd.read_csv(metadata_path)

    # Load Raw JSON
    # We load the entire JSON into memory. Given 220GB RAM, this is safe.
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Prepare arrays
    num_samples = len(df_meta)
    X = np.zeros((num_samples, 3, 75, 75), dtype=np.float32)
    angles = np.zeros(num_samples, dtype=np.float32)
    ids = []
    has_labels = "is_iceberg" in df_meta.columns
    y = np.zeros(num_samples, dtype=np.float32) if has_labels else None

    # Iterate and fill
    # Using original_index to access the list directly is O(1)
    for i, row in df_meta.iterrows():
        orig_idx = int(row["original_index"])
        item = raw_data[orig_idx]

        # Safety check
        if item["id"] != row["id"]:
            raise ValueError(
                f"ID mismatch at index {i}: Meta={row['id']}, JSON={item['id']}"
            )

        # Process Bands
        # Band 1: HH, Band 2: HV
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        # Band 3: Average
        b3 = (b1 + b2) / 2.0

        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

        # Process Angle
        # Metadata CSV already handles 'na' -> NaN conversion
        angles[i] = row["inc_angle"]

        # Process ID
        ids.append(row["id"])

        # Process Label
        if has_labels:
            y[i] = row["is_iceberg"]

    ids = np.array(ids)

    # Save to cache
    np.save(path_X, X)
    np.save(path_angles, angles)
    np.save(path_ids, ids)
    if y is not None:
        np.save(path_y, y)

    return X, angles, y, ids


def get_loaders(batch_size=32, load_cached_data=True, num_workers=2):
    """
    Main function to get DataLoaders. Handles imputation and dataset creation.
    """
    meta_dir = "./metadata"
    input_dir = "./input"

    # 1. Load Data
    X_train, ang_train, y_train, ids_train = process_split(
        os.path.join(meta_dir, "train.csv"),
        os.path.join(input_dir, "train.json"),
        "train",
        load_cached_data,
    )

    X_val, ang_val, y_val, ids_val = process_split(
        os.path.join(meta_dir, "val.csv"),
        os.path.join(input_dir, "train.json"),
        "val",
        load_cached_data,
    )

    X_test, ang_test, y_test, ids_test = process_split(
        os.path.join(meta_dir, "test.csv"),
        os.path.join(input_dir, "test.json"),
        "test",
        load_cached_data,
    )

    # 2. Impute Missing Angles
    # Calculate median from TRAIN set only
    median_angle = np.nanmedian(ang_train)

    # Fill NaNs in all sets using the training median
    ang_train = np.where(np.isnan(ang_train), median_angle, ang_train)
    ang_val = np.where(np.isnan(ang_val), median_angle, ang_val)
    ang_test = np.where(np.isnan(ang_test), median_angle, ang_test)

    # 3. Create Datasets
    train_ds = IcebergDataset(
        X_train, ang_train, ids_train, y_train, transform=get_transforms("train")
    )
    val_ds = IcebergDataset(
        X_val, ang_val, ids_val, y_val, transform=get_transforms("val")
    )
    test_ds = IcebergDataset(
        X_test, ang_test, ids_test, y=None, transform=get_transforms("test")
    )

    # 4. Create DataLoaders
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
