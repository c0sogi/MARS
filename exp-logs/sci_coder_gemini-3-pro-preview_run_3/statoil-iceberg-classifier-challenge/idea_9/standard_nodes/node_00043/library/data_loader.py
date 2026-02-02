import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Iceberg/Ship classification task.
    Handles 3-channel input (HH, HV, Avg) and scalar incidence angle.
    """

    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to tensor
        img = torch.from_numpy(self.X[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float)

        # Apply augmentations (e.g., flips)
        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float)
            return img, angle, label
        else:
            return img, angle


def _process_subset(meta_df, raw_df, is_test=False):
    """
    Helper to extract and process data for a specific subset defined by metadata.
    """
    # Map original_index to positions in raw_df
    indices = meta_df["original_index"].values

    # Extract rows corresponding to the metadata
    subset_raw = raw_df.iloc[indices]

    # Process Band 1 (HH)
    b1 = np.stack(
        subset_raw["band_1"].apply(lambda x: np.array(x).reshape(75, 75)).values
    )

    # Process Band 2 (HV)
    b2 = np.stack(
        subset_raw["band_2"].apply(lambda x: np.array(x).reshape(75, 75)).values
    )

    # Process Band 3 (Average) - proven to help feature extraction
    b3 = (b1 + b2) / 2.0

    # Stack into (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1).astype(np.float32)

    # Extract Angles
    angles = meta_df["inc_angle"].values.astype(np.float32)

    # Extract Labels and IDs
    ids = meta_df["id"].values
    if not is_test:
        y = meta_df["is_iceberg"].values.astype(np.float32)
    else:
        y = None

    return X, angles, y, ids


def load_and_process_data(load_cached_data=True):
    """
    Loads raw data, processes it into numpy arrays, and caches it.
    Imputes missing angles using training set median.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    files = {
        "X_train": "X_train.npy",
        "y_train": "y_train.npy",
        "angles_train": "angles_train.npy",
        "X_val": "X_val.npy",
        "y_val": "y_val.npy",
        "angles_val": "angles_val.npy",
        "X_test": "X_test.npy",
        "ids_test": "ids_test.npy",
        "angles_test": "angles_test.npy",
    }

    # Check if all cache files exist
    cache_exists = all(
        os.path.exists(os.path.join(cache_dir, f)) for f in files.values()
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        data = {}
        for k, v in files.items():
            data[k] = np.load(os.path.join(cache_dir, v), allow_pickle=True)
        return data

    print("Cache not found or disabled. Processing data from scratch...")

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # 2. Impute Incidence Angles
    # Calculate median from TRAIN set only to prevent leakage
    angle_median = train_meta["inc_angle"].median()
    print(f"Imputing missing angles with training median: {angle_median:.4f}")

    train_meta["inc_angle"] = train_meta["inc_angle"].fillna(angle_median)
    val_meta["inc_angle"] = val_meta["inc_angle"].fillna(angle_median)
    test_meta["inc_angle"] = test_meta["inc_angle"].fillna(angle_median)

    # 3. Process Training/Validation Data
    print("Loading train.json...")
    df_train_raw = pd.read_json(Config.TRAIN_JSON)

    print("Processing Train split...")
    X_train, angles_train, y_train, _ = _process_subset(
        train_meta, df_train_raw, is_test=False
    )

    print("Processing Val split...")
    X_val, angles_val, y_val, _ = _process_subset(val_meta, df_train_raw, is_test=False)

    # Free memory
    del df_train_raw

    # 4. Process Test Data
    print("Loading test.json...")
    df_test_raw = pd.read_json(Config.TEST_JSON)

    print("Processing Test split...")
    X_test, angles_test, _, ids_test = _process_subset(
        test_meta, df_test_raw, is_test=True
    )

    del df_test_raw

    # 5. Save to Cache
    print("Saving processed data to cache...")
    np.save(os.path.join(cache_dir, files["X_train"]), X_train)
    np.save(os.path.join(cache_dir, files["y_train"]), y_train)
    np.save(os.path.join(cache_dir, files["angles_train"]), angles_train)

    np.save(os.path.join(cache_dir, files["X_val"]), X_val)
    np.save(os.path.join(cache_dir, files["y_val"]), y_val)
    np.save(os.path.join(cache_dir, files["angles_val"]), angles_val)

    np.save(os.path.join(cache_dir, files["X_test"]), X_test)
    np.save(os.path.join(cache_dir, files["ids_test"]), ids_test)
    np.save(os.path.join(cache_dir, files["angles_test"]), angles_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angles_train": angles_train,
        "X_val": X_val,
        "y_val": y_val,
        "angles_val": angles_val,
        "X_test": X_test,
        "ids_test": ids_test,
        "angles_test": angles_test,
    }


def get_loaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    Applies augmentations to the training set.
    """
    data = load_and_process_data(load_cached_data=load_cached_data)

    # Augmentations for training: Random flips
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Instantiate Datasets
    train_dataset = IcebergDataset(
        data["X_train"],
        data["angles_train"],
        data["y_train"],
        transform=train_transform,
    )

    val_dataset = IcebergDataset(
        data["X_val"], data["angles_val"], data["y_val"], transform=None
    )

    test_dataset = IcebergDataset(
        data["X_test"], data["angles_test"], y=None, transform=None
    )

    # Debugging: Subset datasets if requested
    if debug:
        subset_size = 64
        print(f"DEBUG mode: Reducing dataset size to {subset_size}")
        train_dataset = Subset(
            train_dataset, range(min(len(train_dataset), subset_size))
        )
        val_dataset = Subset(val_dataset, range(min(len(val_dataset), subset_size)))
        test_dataset = Subset(test_dataset, range(min(len(test_dataset), subset_size)))
        # Note: ids_test will still be full length, caller must handle if strictly needed,
        # but for debug training loop it's fine.

    # Create DataLoaders
    # Pin memory enables faster data transfer to CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader, data["ids_test"]
