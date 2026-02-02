import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), float32.
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), float32.
            ids (np.ndarray, optional): Shape (N,), string/object.
            transform (callable, optional): Transform to apply to the image tensor.
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
        img = self.images[idx]  # (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms (augmentation)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        sample = {"image": img_tensor, "angle": angle_tensor}

        if self.labels is not None:
            label = self.labels[idx]
            sample["label"] = torch.tensor(label, dtype=torch.float32)

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def _load_raw_json(json_path):
    """Loads raw JSON and creates a lookup dictionary by ID."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def _process_split(metadata_df, raw_data_dict, angle_impute_val, is_test=False):
    """
    Converts raw data into numpy arrays based on metadata.
    Constructs 3-channel images: HH, HV, Avg(HH, HV).
    """
    num_samples = len(metadata_df)
    X = np.zeros((num_samples, 3, 75, 75), dtype=np.float32)
    angles = np.zeros(num_samples, dtype=np.float32)
    ids = []
    y = np.zeros(num_samples, dtype=np.float32) if not is_test else None

    meta_ids = metadata_df["id"].values
    meta_angles = metadata_df["inc_angle"].values
    meta_labels = metadata_df["is_iceberg"].values if not is_test else None

    for i in range(num_samples):
        img_id = meta_ids[i]
        item = raw_data_dict[img_id]

        # Process Bands
        # Band 1 (HH)
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        # Band 2 (HV)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        # Band 3 (Avg)
        b3 = (b1 + b2) / 2.0

        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

        # Process Angle
        ang = meta_angles[i]
        if np.isnan(ang):
            if angle_impute_val is None:
                # Fallback, though impute val should always be provided
                angles[i] = 0.0
            else:
                angles[i] = angle_impute_val
        else:
            angles[i] = ang

        ids.append(img_id)

        if not is_test:
            y[i] = meta_labels[i]

    ids = np.array(ids)
    return X, angles, y, ids


def get_data(
    mode,
    metadata_path,
    raw_json_path,
    cache_dir,
    load_cached_data,
    angle_impute_val=None,
):
    """
    Generic function to load/process data for a specific split (train/val/test) with caching.
    """
    # Define cache filenames
    cache_X = os.path.join(cache_dir, f"X_{mode}.npy")
    cache_angles = os.path.join(cache_dir, f"angle_{mode}.npy")
    cache_ids = os.path.join(cache_dir, f"ids_{mode}.npy")
    cache_y = os.path.join(cache_dir, f"y_{mode}.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        has_X = os.path.exists(cache_X)
        has_ang = os.path.exists(cache_angles)
        has_ids = os.path.exists(cache_ids)
        has_y = os.path.exists(cache_y) if mode != "test" else True

        if has_X and has_ang and has_ids and has_y:
            X = np.load(cache_X)
            angles = np.load(cache_angles)
            ids = np.load(cache_ids)
            y = np.load(cache_y) if mode != "test" else None
            return X, angles, y, ids

    # 2. Process from Scratch
    df_meta = pd.read_csv(metadata_path)
    raw_data_dict = _load_raw_json(raw_json_path)

    X, angles, y, ids = _process_split(
        df_meta, raw_data_dict, angle_impute_val, is_test=(mode == "test")
    )

    # 3. Save to Cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_X, X)
    np.save(cache_angles, angles)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, angles, y, ids


def get_dataloaders(
    input_dir="./input",
    metadata_dir="./metadata",
    cache_dir="./working/idea_50",
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
    seed=42,
):
    """
    Main function to get data loaders.
    """
    set_seed(seed)

    # 1. Determine Angle Imputation Value (Median of Train)
    train_meta_path = os.path.join(metadata_dir, "train.csv")
    df_train = pd.read_csv(train_meta_path)
    angle_impute_val = df_train["inc_angle"].median()

    # 2. Load Data Splits
    # Train
    X_train, ang_train, y_train, ids_train = get_data(
        "train",
        train_meta_path,
        os.path.join(input_dir, "train.json"),
        cache_dir,
        load_cached_data,
        angle_impute_val,
    )

    # Val
    X_val, ang_val, y_val, ids_val = get_data(
        "val",
        os.path.join(metadata_dir, "val.csv"),
        os.path.join(input_dir, "train.json"),
        cache_dir,
        load_cached_data,
        angle_impute_val,
    )

    # Test
    X_test, ang_test, y_test, ids_test = get_data(
        "test",
        os.path.join(metadata_dir, "test.csv"),
        os.path.join(input_dir, "test.json"),
        cache_dir,
        load_cached_data,
        angle_impute_val,
    )

    # 3. Define Transforms
    # Train: Random Flip
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Val/Test: None
    val_transform = None

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val, ang_val, y_val, ids_val, transform=val_transform
    )
    test_dataset = IcebergDataset(
        X_test, ang_test, ids=ids_test, transform=val_transform
    )

    # 5. Create DataLoaders
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

    return train_loader, val_loader, test_loader
