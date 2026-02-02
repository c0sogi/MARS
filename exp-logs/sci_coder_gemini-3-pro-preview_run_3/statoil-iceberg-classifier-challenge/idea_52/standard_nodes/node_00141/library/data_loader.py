import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Ship vs Iceberg classification.
    Constructs a 4-channel input: HH, HV, Avg(HH, HV), Ratio(HH, HV).
    """

    def __init__(self, X, y, angle, transform=False):
        self.X = X
        self.y = y
        self.angle = angle
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve 2-channel image (HH, HV)
        # Shape: (2, 75, 75)
        img = self.X[idx]

        hh = img[0]
        hv = img[1]

        # Compute synthetic bands
        # Synthetic Average: (HH + HV) / 2
        avg = (hh + hv) / 2.0
        # Depolarization Ratio: HH - HV
        ratio = hh - hv

        # Stack to create 4-channel input
        # Shape: (4, 75, 75)
        img_4ch = np.stack([hh, hv, avg, ratio], axis=0)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_4ch).float()
        ang_tensor = torch.tensor([self.angle[idx]], dtype=torch.float32)

        # Apply Augmentations
        if self.transform:
            # Random Horizontal Flip
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
            # Random Vertical Flip
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [1])

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, ang_tensor, label
        else:
            return img_tensor, ang_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches it.
    Implements median imputation for incidence angles.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if cache exists
    if load_cached_data and all(os.path.exists(f) for f in files.values()):
        print("Loading cached data...")
        data = {k: np.load(v, allow_pickle=True) for k, v in files.items()}
        return data

    print("Processing raw data from scratch...")

    # --- Process Train ---
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    df_train = pd.DataFrame(train_data)

    # Images: Reshape flattened 5625 -> 75x75
    # We store as 2 channels (HH, HV) to save space; expansion happens in Dataset
    b1_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_1"]
        ]
    )
    b2_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_2"]
        ]
    )
    X_train = np.stack([b1_train, b2_train], axis=1)  # (N, 2, 75, 75)

    # Targets
    y_train = df_train["is_iceberg"].values.astype(np.float32)

    # Angles
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    angle_train = df_train["inc_angle"].values.astype(np.float32)

    # --- Process Test ---
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_test = pd.DataFrame(test_data)

    b1_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_1"]
        ]
    )
    b2_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_2"]
        ]
    )
    X_test = np.stack([b1_test, b2_test], axis=1)

    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    angle_test = df_test["inc_angle"].values.astype(np.float32)
    ids_test = df_test["id"].values

    # --- Imputation ---
    # Combine train and test angles to compute global median for robustness
    all_angles = np.concatenate([angle_train, angle_test])
    median_angle = np.nanmedian(all_angles)

    angle_train[np.isnan(angle_train)] = median_angle
    angle_test[np.isnan(angle_test)] = median_angle

    # --- Save to Cache ---
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["angle_train"], angle_train)
    np.save(files["X_test"], X_test)
    np.save(files["angle_test"], angle_test)
    np.save(files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angle_train": angle_train,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }


def get_dataloaders(data, fold_idx=0):
    """
    Creates Stratified K-Fold DataLoaders for the specified fold.
    """
    X = data["X_train"]
    y = data["y_train"]
    angle = data["angle_train"]

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    splits = list(skf.split(X, y))
    if fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Subset data
    X_train, y_train, angle_train = X[train_idx], y[train_idx], angle[train_idx]
    X_val, y_val, angle_val = X[val_idx], y[val_idx], angle[val_idx]

    # Debug Mode: Reduce dataset size
    if Config.DEBUG:
        limit = Config.DEBUG_SUBSET_SIZE
        X_train, y_train, angle_train = (
            X_train[:limit],
            y_train[:limit],
            angle_train[:limit],
        )
        X_val, y_val, angle_val = X_val[:limit], y_val[:limit], angle_val[:limit]

    # Create Datasets
    # Train set gets transformations (flips)
    train_dataset = IcebergDataset(X_train, y_train, angle_train, transform=True)
    # Validation set has no transformations
    val_dataset = IcebergDataset(X_val, y_val, angle_val, transform=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
