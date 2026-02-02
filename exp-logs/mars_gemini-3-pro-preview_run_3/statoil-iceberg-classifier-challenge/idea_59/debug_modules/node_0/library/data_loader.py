import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import seed_everything, get_logger

logger = get_logger("data_loader")


def process_images(df):
    """
    Converts flattened band data into 3-channel image arrays (HH, HV, Avg).

    Args:
        df (pd.DataFrame): DataFrame containing 'band_1' and 'band_2' columns.

    Returns:
        np.ndarray: Array of shape (N, 3, 75, 75).
    """
    # Band 1 and Band 2 are lists of floats
    b1 = np.array(df["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(df["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    # Synthetic 3rd channel (Average)
    b3 = (b1 + b2) / 2.0

    # Stack: (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1)
    return X


def get_data(load_cached_data=True):
    """
    Loads data from JSON or Cache.
    Returns a dictionary containing numpy arrays for X, y, angles, and ids.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy cache first.

    Returns:
        dict: Dictionary with keys 'X_train', 'y_train', 'angles_train', etc.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(Config.WORK_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.WORK_DIR, "y_train.npy"),
        "angles_train": os.path.join(Config.WORK_DIR, "angles_train.npy"),
        "ids_train": os.path.join(Config.WORK_DIR, "ids_train.npy"),
        "X_test": os.path.join(Config.WORK_DIR, "X_test.npy"),
        "angles_test": os.path.join(Config.WORK_DIR, "angles_test.npy"),
        "ids_test": os.path.join(Config.WORK_DIR, "ids_test.npy"),
    }

    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        logger.info("Loading data from cache...")
        data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}
        return data

    logger.info("Processing data from scratch...")

    # Load Raw Data
    train_path = os.path.join(Config.INPUT_DIR, "train.json")
    test_path = os.path.join(Config.INPUT_DIR, "test.json")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Input files not found in {Config.INPUT_DIR}")

    df_train = pd.read_json(train_path)
    df_test = pd.read_json(test_path)

    # Process Train
    X_train = process_images(df_train)
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    # Coerce 'na' to NaN, then convert to float
    angles_train = pd.to_numeric(df_train["inc_angle"], errors="coerce").values.astype(
        np.float32
    )
    ids_train = df_train["id"].values

    # Process Test
    X_test = process_images(df_test)
    angles_test = pd.to_numeric(df_test["inc_angle"], errors="coerce").values.astype(
        np.float32
    )
    ids_test = df_test["id"].values

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angles_train": angles_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angles_test": angles_test,
        "ids_test": ids_test,
    }


class IcebergDataset(Dataset):
    def __init__(
        self,
        X,
        angles,
        y=None,
        transform=None,
        angle_scaler=None,
        angle_imputer_val=None,
    ):
        """
        Dataset class that handles image and angle retrieval.

        Args:
            X (np.ndarray): Image data.
            angles (np.ndarray): Incidence angles (may contain NaNs).
            y (np.ndarray, optional): Labels.
            transform (callable, optional): Augmentation function.
            angle_scaler (StandardScaler, optional): Scaler fitted on training data.
            angle_imputer_val (float, optional): Value to replace NaNs (median of training data).
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.transform = transform

        # Handle Angles
        angles_np = np.array(angles).flatten()

        # Imputation
        if angle_imputer_val is not None:
            # Fill NaNs with the provided value (calculated from training set)
            self.raw_angles = np.nan_to_num(angles_np, nan=angle_imputer_val)
        else:
            # Fallback if no imputer provided (should be avoided in pipeline)
            self.raw_angles = np.nan_to_num(angles_np, nan=0.0)

        # Normalization for AC-SE
        if angle_scaler is not None:
            # Reshape for scaler (N, 1) then flatten back
            self.norm_angles = angle_scaler.transform(
                self.raw_angles.reshape(-1, 1)
            ).flatten()
        else:
            self.norm_angles = self.raw_angles

        # Convert to tensors
        self.raw_angles = torch.FloatTensor(self.raw_angles)
        self.norm_angles = torch.FloatTensor(self.norm_angles)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        raw_ang = self.raw_angles[idx]
        norm_ang = self.norm_angles[idx]

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            return img, raw_ang, norm_ang, self.y[idx]
        else:
            return img, raw_ang, norm_ang


class TrainTransform:
    """Simple augmentation: Horizontal and Vertical Flips"""

    def __call__(self, x):
        # x is (3, 75, 75) tensor
        if np.random.rand() > 0.5:
            x = torch.flip(x, [2])  # Horizontal flip (width dim is last)
        if np.random.rand() > 0.5:
            x = torch.flip(x, [1])  # Vertical flip (height dim is second last)
        return x


def get_fold_loaders(fold_idx, data, batch_size=Config.BATCH_SIZE, num_workers=2):
    """
    Creates train and validation loaders for a specific fold, ensuring leak-free preprocessing.

    Args:
        fold_idx (int): Index of the current fold (0 to NUM_FOLDS-1).
        data (dict): Data dictionary from get_data().
        batch_size (int): Batch size.
        num_workers (int): Number of dataloader workers.

    Returns:
        tuple: (train_loader, val_loader, scaler, imputation_val)
    """
    X = data["X_train"]
    y = data["y_train"]
    angles = data["angles_train"]

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for all folds
    splits = list(skf.split(X, y))
    if fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Split Data
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    ang_tr, ang_val = angles[train_idx], angles[val_idx]

    # --- LEAK-FREE PREPROCESSING ---
    # 1. Calculate Imputation Value (Median) ONLY on Training Data
    tr_median = np.nanmedian(ang_tr)

    # 2. Fit Scaler ONLY on Training Data (after imputation)
    # We impute temporarily for fitting the scaler
    ang_tr_filled_temp = np.nan_to_num(ang_tr, nan=tr_median)
    scaler = StandardScaler()
    scaler.fit(ang_tr_filled_temp.reshape(-1, 1))

    # Create Datasets
    # Pass raw angles + imputation value + scaler.
    # The dataset class handles applying imputation and normalization internally.
    train_ds = IcebergDataset(
        X_tr,
        ang_tr,
        y_tr,
        transform=TrainTransform(),
        angle_scaler=scaler,
        angle_imputer_val=tr_median,
    )

    val_ds = IcebergDataset(
        X_val,
        ang_val,
        y_val,
        transform=None,
        angle_scaler=scaler,
        angle_imputer_val=tr_median,
    )

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

    return train_loader, val_loader, scaler, tr_median


def get_test_loader(
    data, scaler, imputation_val, batch_size=Config.BATCH_SIZE, num_workers=2
):
    """
    Creates test loader using preprocessing statistics from a specific fold.

    Args:
        data (dict): Data dictionary from get_data().
        scaler (StandardScaler): Fitted scaler from the training fold.
        imputation_val (float): Median angle from the training fold.
        batch_size (int): Batch size.
        num_workers (int): Number of dataloader workers.

    Returns:
        DataLoader: Test data loader.
    """
    X_test = data["X_test"]
    angles_test = data["angles_test"]

    test_ds = IcebergDataset(
        X_test,
        angles_test,
        y=None,
        transform=None,
        angle_scaler=scaler,
        angle_imputer_val=imputation_val,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
