import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.utils import seed_everything

# Configuration
CACHE_DIR = "./working/idea_18/"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel SAR images and incidence angles.
    Applies on-the-fly augmentation for training.
    """

    def __init__(self, images, angles, labels=None, transform=False):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        x_img = torch.from_numpy(img).float()
        x_angle = torch.tensor([angle], dtype=torch.float32)

        # Apply Augmentation (Training only)
        if self.transform:
            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            x_img = torch.rot90(x_img, k, dims=[1, 2])

            # Random Horizontal Flip
            if np.random.random() > 0.5:
                x_img = torch.flip(x_img, dims=[2])

        if self.labels is not None:
            y = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return (x_img, x_angle), y
        else:
            return (x_img, x_angle)


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes bands, normalizes, and caches the result.

    Returns:
        train_data: dict containing 'images', 'angles', 'labels'
        test_data: dict containing 'images', 'angles', 'ids'
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return (
            {
                "images": data["train_images"],
                "angles": data["train_angles"],
                "labels": data["train_labels"],
            },
            {
                "images": data["test_images"],
                "angles": data["test_angles"],
                "ids": data["test_ids"],
            },
        )

    print("Processing data from scratch...")

    # 1. Load Metadata to identify splits (though we combine train/val for CV)
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    # Combine to form full training set for CV
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)
    train_ids = set(full_train_meta["id"].values)

    # 2. Load Raw JSON
    with open(os.path.join(INPUT_DIR, "train.json"), "r") as f:
        raw_train = json.load(f)
    with open(os.path.join(INPUT_DIR, "test.json"), "r") as f:
        raw_test = json.load(f)

    # Filter train data to match metadata (safety check)
    train_data_list = [item for item in raw_train if item["id"] in train_ids]

    # 3. Helper to process list of dicts into arrays
    def extract_features(data_list, is_test=False):
        ids = []
        band_1 = []
        band_2 = []
        angles = []
        labels = []

        for item in data_list:
            ids.append(item["id"])
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            band_1.append(b1)
            band_2.append(b2)

            # Handle incidence angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            if not is_test:
                labels.append(item["is_iceberg"])

        return (
            np.array(ids),
            np.stack(band_1),
            np.stack(band_2),
            np.array(angles),
            np.array(labels) if not is_test else None,
        )

    # Extract raw arrays
    print("Extracting features...")
    tr_ids, tr_b1, tr_b2, tr_angles, tr_labels = extract_features(
        train_data_list, is_test=False
    )
    te_ids, te_b1, te_b2, te_angles, _ = extract_features(raw_test, is_test=True)

    # 4. Impute Incidence Angles
    # Calculate mean from training set (ignoring NaNs)
    angle_mean = np.nanmean(tr_angles)

    # Fill NaNs
    tr_angles = np.nan_to_num(tr_angles, nan=angle_mean)
    te_angles = np.nan_to_num(te_angles, nan=angle_mean)

    # 5. Construct 3-Channel Images: [Band1, Band2, Avg]
    print("Constructing 3-channel images...")
    tr_b3 = (tr_b1 + tr_b2) / 2.0
    te_b3 = (te_b1 + te_b2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    # Note: PyTorch expects (C, H, W).
    # np.stack usually does (N, H, W). We stack along axis 1.
    tr_images = np.stack([tr_b1, tr_b2, tr_b3], axis=1)
    te_images = np.stack([te_b1, te_b2, te_b3], axis=1)

    # 6. Normalization (Independent Per-Channel Min-Max)
    print("Normalizing data...")
    for c in range(3):
        # Calculate stats on training set only
        c_min = tr_images[:, c, :, :].min()
        c_max = tr_images[:, c, :, :].max()

        # Apply to train
        tr_images[:, c, :, :] = (tr_images[:, c, :, :] - c_min) / (c_max - c_min)

        # Apply to test
        te_images[:, c, :, :] = (te_images[:, c, :, :] - c_min) / (c_max - c_min)

    # 7. Cache Data
    print(f"Saving processed data to {cache_path}...")
    np.savez(
        cache_path,
        train_images=tr_images,
        train_angles=tr_angles,
        train_labels=tr_labels,
        test_images=te_images,
        test_angles=te_angles,
        test_ids=te_ids,
    )

    return (
        {"images": tr_images, "angles": tr_angles, "labels": tr_labels},
        {"images": te_images, "angles": te_angles, "ids": te_ids},
    )


def get_folds(train_data, n_splits=5, seed=42):
    """
    Generates Stratified K-Folds.

    Args:
        train_data (dict): Dictionary containing 'images', 'angles', 'labels'.
        n_splits (int): Number of folds.
        seed (int): Random seed.

    Returns:
        list of tuples: [(train_indices, val_indices), ...]
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = train_data["labels"]
    # X is dummy, we only need indices
    folds = list(skf.split(np.zeros(len(y)), y))
    return folds


def get_dataloaders(fold_idx, folds, train_data, batch_size=32, num_workers=2):
    """
    Creates DataLoaders for a specific fold.

    Args:
        fold_idx (int): Index of the fold to use (0 to n_splits-1).
        folds (list): List of (train_idx, val_idx) tuples.
        train_data (dict): Dictionary with keys 'images', 'angles', 'labels'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        train_loader, val_loader
    """
    train_idx, val_idx = folds[fold_idx]

    # Split data
    X_train = train_data["images"][train_idx]
    inc_train = train_data["angles"][train_idx]
    y_train = train_data["labels"][train_idx]

    X_val = train_data["images"][val_idx]
    inc_val = train_data["angles"][val_idx]
    y_val = train_data["labels"][val_idx]

    # Create Datasets
    # Enable transform (augmentation) for training
    train_dataset = IcebergDataset(X_train, inc_train, y_train, transform=True)
    # Disable transform for validation
    val_dataset = IcebergDataset(X_val, inc_val, y_val, transform=False)

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

    return train_loader, val_loader


def get_test_loader(test_data, batch_size=32, num_workers=2):
    """
    Creates DataLoader for the test set.

    Args:
        test_data (dict): Dictionary with keys 'images', 'angles', 'ids'.

    Returns:
        test_loader
    """
    test_dataset = IcebergDataset(
        test_data["images"], test_data["angles"], labels=None, transform=False
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
