import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config, set_seed


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
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
        # Convert numpy array to torch tensor
        # Input X is (3, 75, 75) float32
        img = torch.from_numpy(self.X[idx])
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply augmentations if any
        if self.transform:
            img = self.transform(img)

        # Return data based on mode (Train/Val vs Test)
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            # For test set, return ID to map predictions
            id_val = self.ids[idx]
            return img, angle, id_val


def load_and_process_data(load_cached_data=True):
    """
    Loads raw data, processes it (reshaping, imputation), and caches it.
    Returns processed numpy arrays for Train, Val, and Test sets.
    """
    # Define cache file paths
    cache_files = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        Config.CACHE_TRAIN_ANGLE,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        Config.CACHE_VAL_ANGLE,
        Config.CACHE_TEST_X,
        Config.CACHE_TEST_IDS,
        Config.CACHE_TEST_ANGLE,
    ]

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print(f"[{Config.PROJECT_NAME}] Loading data from cache...")
        X_train = np.load(Config.CACHE_TRAIN_X)
        y_train = np.load(Config.CACHE_TRAIN_Y)
        angle_train = np.load(Config.CACHE_TRAIN_ANGLE)

        X_val = np.load(Config.CACHE_VAL_X)
        y_val = np.load(Config.CACHE_VAL_Y)
        angle_val = np.load(Config.CACHE_VAL_ANGLE)

        X_test = np.load(Config.CACHE_TEST_X)
        ids_test = np.load(Config.CACHE_TEST_IDS)
        angle_test = np.load(Config.CACHE_TEST_ANGLE)

        return (
            (X_train, y_train, angle_train),
            (X_val, y_val, angle_val),
            (X_test, ids_test, angle_test),
        )

    print(f"[{Config.PROJECT_NAME}] Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw JSON Data
    # Note: Loading entire JSON into memory is feasible given 220GB RAM
    print(f"[{Config.PROJECT_NAME}] Loading raw train.json...")
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train = json.load(f)

    print(f"[{Config.PROJECT_NAME}] Loading raw test.json...")
    with open(Config.TEST_JSON, "r") as f:
        raw_test = json.load(f)

    # Helper function to extract and process samples
    def process_subset(meta_df, raw_data_list, is_test=False):
        X_list = []
        angle_list = []
        y_list = []
        id_list = []

        for _, row in meta_df.iterrows():
            # Retrieve raw item using original_index
            idx = row["original_index"]
            item = raw_data_list[idx]

            # Verify ID match (sanity check)
            if item["id"] != row["id"]:
                raise ValueError(
                    f"ID mismatch at index {idx}: Meta {row['id']} vs Raw {item['id']}"
                )

            # Extract Bands
            # Raw data is flattened 5625 list
            band_1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            band_2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

            # Create 3rd Channel (Average)
            band_avg = (band_1 + band_2) / 2.0

            # Stack to (3, 75, 75)
            img = np.stack([band_1, band_2, band_avg], axis=0)
            X_list.append(img)

            # Extract Angle
            ang = item["inc_angle"]
            if ang == "na":
                angle_list.append(np.nan)
            else:
                angle_list.append(float(ang))

            # Extract ID
            id_list.append(item["id"])

            # Extract Target
            if not is_test:
                y_list.append(item["is_iceberg"])

        X_arr = np.array(X_list, dtype=np.float32)
        angle_arr = np.array(angle_list, dtype=np.float32)
        id_arr = np.array(id_list)
        y_arr = np.array(y_list, dtype=np.float32) if not is_test else None

        return X_arr, angle_arr, y_arr, id_arr

    # Process Splits
    print(f"[{Config.PROJECT_NAME}] constructing arrays...")
    X_train, angle_train, y_train, _ = process_subset(
        train_meta, raw_train, is_test=False
    )
    X_val, angle_val, y_val, _ = process_subset(val_meta, raw_train, is_test=False)
    X_test, angle_test, _, ids_test = process_subset(test_meta, raw_test, is_test=True)

    # Impute Missing Angles
    # Calculate median from Train set (ignoring NaNs)
    median_angle = np.nanmedian(angle_train)
    print(
        f"[{Config.PROJECT_NAME}] Imputing missing angles with median: {median_angle:.4f}"
    )

    # Fill NaNs
    # Note: We use the same training median for Val and Test to prevent leakage
    angle_train[np.isnan(angle_train)] = median_angle
    angle_val[np.isnan(angle_val)] = median_angle
    angle_test[np.isnan(angle_test)] = median_angle

    # Save to Cache
    print(f"[{Config.PROJECT_NAME}] Saving to cache...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(Config.CACHE_TRAIN_X, X_train)
    np.save(Config.CACHE_TRAIN_Y, y_train)
    np.save(Config.CACHE_TRAIN_ANGLE, angle_train)

    np.save(Config.CACHE_VAL_X, X_val)
    np.save(Config.CACHE_VAL_Y, y_val)
    np.save(Config.CACHE_VAL_ANGLE, angle_val)

    np.save(Config.CACHE_TEST_X, X_test)
    np.save(Config.CACHE_TEST_IDS, ids_test)
    np.save(Config.CACHE_TEST_ANGLE, angle_test)

    return (
        (X_train, y_train, angle_train),
        (X_val, y_val, angle_val),
        (X_test, ids_test, angle_test),
    )


def get_loaders(fold=None, load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        fold (int, optional): If provided, performs K-Fold split on the merged Train+Val data.
                              If None, uses the fixed split from metadata.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Data
    train_data, val_data, test_data = load_and_process_data(load_cached_data)

    X_train_fixed, y_train_fixed, angle_train_fixed = train_data
    X_val_fixed, y_val_fixed, angle_val_fixed = val_data
    X_test, ids_test, angle_test = test_data

    # Determine Train/Val split based on fold
    if fold is not None:
        # Merge fixed train and val to perform CV
        X_total = np.concatenate([X_train_fixed, X_val_fixed], axis=0)
        y_total = np.concatenate([y_train_fixed, y_val_fixed], axis=0)
        angle_total = np.concatenate([angle_train_fixed, angle_val_fixed], axis=0)

        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Get indices for the specific fold
        # We iterate to find the specific fold indices
        fold_generator = skf.split(X_total, y_total)
        for i, (train_idx, val_idx) in enumerate(fold_generator):
            if i == fold:
                X_train = X_total[train_idx]
                y_train = y_total[train_idx]
                angle_train = angle_total[train_idx]

                X_val = X_total[val_idx]
                y_val = y_total[val_idx]
                angle_val = angle_total[val_idx]
                break
        else:
            raise ValueError(f"Fold {fold} out of range for {Config.NUM_FOLDS} splits.")

    else:
        # Use fixed splits
        X_train, y_train, angle_train = X_train_fixed, y_train_fixed, angle_train_fixed
        X_val, y_val, angle_val = X_val_fixed, y_val_fixed, angle_val_fixed

    # Define Transforms
    # Note: Input is (3, 75, 75) Tensor. RandomHorizontalFlip works on (C, H, W).
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # No TTA or transforms for validation/test in this loader
    val_transform = None
    test_transform = None

    # Create Datasets
    train_ds = IcebergDataset(X_train, angle_train, y_train, transform=train_transform)
    val_ds = IcebergDataset(X_val, angle_val, y_val, transform=val_transform)
    test_ds = IcebergDataset(X_test, angle_test, ids=ids_test, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
