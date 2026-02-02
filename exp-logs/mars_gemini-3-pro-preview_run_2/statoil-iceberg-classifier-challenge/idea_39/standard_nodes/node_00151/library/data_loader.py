import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class GlobalMinMaxScaler:
    """
    Scales images using global min and max statistics derived from the entire training set.
    This prevents covariate shift caused by fold-wise or sample-wise scaling.
    """

    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, images):
        """
        Compute global min and max for each channel across the entire dataset.
        images: (N, H, W, C)
        """
        # Reshape to (N*H*W, C) to compute stats per channel
        reshaped = images.reshape(-1, images.shape[-1])
        self.min_vals = np.min(reshaped, axis=0)
        self.max_vals = np.max(reshaped, axis=0)

    def transform(self, images):
        """
        Apply min-max scaling: (x - min) / (max - min).
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Scaler not fitted.")

        # Broadcasting (1, 1, 1, C)
        min_v = self.min_vals.reshape(1, 1, 1, -1)
        max_v = self.max_vals.reshape(1, 1, 1, -1)

        # Avoid division by zero
        denom = max_v - min_v
        denom[denom == 0] = 1.0

        return (images - min_v) / denom


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into 3-channel images, applies global scaling,
    and caches the result to disk.

    Returns a dictionary containing processed arrays.
    """
    cache_path = Config.CACHE_FILE

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "X_train": data["X_train"],
                "y_train": data["y_train"],
                "angles_train": data["angles_train"],
                "ids_train": data["ids_train"],
                "X_test": data["X_test"],
                "angles_test": data["angles_test"],
                "ids_test": data["ids_test"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch

    # Load raw JSON
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Helper to process a list of dicts into arrays
    def extract_features(data_list, is_train=True):
        ids = []
        band_1 = []
        band_2 = []
        angles = []
        labels = []

        for item in data_list:
            ids.append(item["id"])
            # Reshape flattened bands
            b1 = np.array(item["band_1"]).reshape(Config.IMG_HEIGHT, Config.IMG_WIDTH)
            b2 = np.array(item["band_2"]).reshape(Config.IMG_HEIGHT, Config.IMG_WIDTH)
            band_1.append(b1)
            band_2.append(b2)

            # Process Incidence Angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            if is_train:
                labels.append(item["is_iceberg"])

        # Stack bands
        b1_stack = np.stack(band_1, axis=0)  # (N, 75, 75)
        b2_stack = np.stack(band_2, axis=0)

        # Create 3rd channel: Mean of Band 1 and Band 2
        b3_stack = (b1_stack + b2_stack) / 2.0

        # Stack channels: (N, 75, 75, 3)
        X = np.stack([b1_stack, b2_stack, b3_stack], axis=-1)

        angles = np.array(angles, dtype=np.float32)
        ids = np.array(ids)

        if is_train:
            y = np.array(labels, dtype=np.float32)
            return X, angles, ids, y
        else:
            return X, angles, ids, None

    # Extract features
    X_train_raw, angles_train, ids_train, y_train = extract_features(
        train_data, is_train=True
    )
    X_test_raw, angles_test, ids_test, _ = extract_features(test_data, is_train=False)

    # Impute missing angles in training set (replace NaN with mean of valid training angles)
    valid_angles = angles_train[~np.isnan(angles_train)]
    angle_mean = np.mean(valid_angles)
    angles_train[np.isnan(angles_train)] = angle_mean
    # Apply same mean to test set if needed
    angles_test[np.isnan(angles_test)] = angle_mean

    # Global Scaling
    # Fit on training data only to avoid leakage
    scaler = GlobalMinMaxScaler()
    scaler.fit(X_train_raw)

    # Transform both datasets
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=X_train_scaled,
        y_train=y_train,
        angles_train=angles_train,
        ids_train=ids_train,
        X_test=X_test_scaled,
        angles_test=angles_test,
        ids_test=ids_test,
    )

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "angles_train": angles_train,
        "ids_train": ids_train,
        "X_test": X_test_scaled,
        "angles_test": angles_test,
        "ids_test": ids_test,
    }


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=False):
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (75, 75, 3)
        angle = self.angles[idx]

        # Augmentation
        if self.transform:
            # Random Rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            img = np.rot90(img, k=k, axes=(0, 1))

            # Horizontal Flip
            if Config.AUG_H_FLIP and np.random.random() > 0.5:
                img = np.fliplr(img)

            # Vertical Flip (Disabled in Config, but logic included for completeness)
            if Config.AUG_V_FLIP and np.random.random() > 0.5:
                img = np.flipud(img)

        # Convert to Tensor (C, H, W)
        # img is currently (H, W, C) -> transpose to (C, H, W)
        # Use .copy() to handle negative strides from rot90/flip
        img = np.transpose(img, (2, 0, 1)).copy()
        img_tensor = torch.from_numpy(img).float()

        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor, torch.tensor(-1.0)  # Dummy label for test


def get_loaders(fold_idx=None, load_cached_data=True):
    """
    Returns train_loader, val_loader, test_loader.

    Args:
        fold_idx (int, optional): The fold index (0-4) for Stratified K-Fold.
                                  If None, defaults to Fold 0.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    data = process_data(load_cached_data)

    X_full = data["X_train"]
    y_full = data["y_train"]
    angles_full = data["angles_train"]

    X_test = data["X_test"]
    angles_test = data["angles_test"]

    # Determine Train/Val indices using Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(X_full, y_full))

    current_fold = fold_idx if fold_idx is not None else 0
    train_idx, val_idx = splits[current_fold]

    # Create subsets
    X_train, y_train, ang_train = (
        X_full[train_idx],
        y_full[train_idx],
        angles_full[train_idx],
    )
    X_val, y_val, ang_val = X_full[val_idx], y_full[val_idx], angles_full[val_idx]

    # Create Datasets
    # Apply augmentation only to training set
    train_ds = IcebergDataset(X_train, ang_train, y_train, transform=True)
    val_ds = IcebergDataset(X_val, ang_val, y_val, transform=False)
    test_ds = IcebergDataset(X_test, angles_test, y=None, transform=False)

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
