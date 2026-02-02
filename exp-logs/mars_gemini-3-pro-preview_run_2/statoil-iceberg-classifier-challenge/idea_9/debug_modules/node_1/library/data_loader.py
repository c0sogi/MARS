import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel SAR images, incidence angles, and labels.
    """

    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,) or (N, 1)
            labels (np.ndarray, optional): Shape (N,) or (N, 1)
            transform (bool): Whether to apply data augmentation (rotation/flip).
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx]  # (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply Augmentations (only for training)
        if self.transform:
            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])

            # Random Horizontal Flip
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])

        # Handle Label
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # Return 0.0 as dummy label for test set
            return img_tensor, angle_tensor, torch.tensor(0.0, dtype=torch.float32)


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into 3-channel images, handles missing values,
    normalizes the data, and caches the result.

    Returns:
        dict: Contains processed arrays and ID mappings.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        try:
            print(f"Loading cached data from {Config.PROCESSED_DATA_PATH}...")
            cached = np.load(Config.PROCESSED_DATA_PATH, allow_pickle=True)
            return {key: cached[key] for key in cached.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    print("Processing data from raw JSON files...")

    # 2. Load Raw Data
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Convert to DataFrames for easier handling
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # 3. Process Incidence Angles
    # Replace 'na' with NaN and impute with mean of training set
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")

    mean_angle = df_train["inc_angle"].mean()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(mean_angle)
    # Note: Test set usually shouldn't have missing values based on description, but we handle it safely
    df_test["inc_angle"] = df_test["inc_angle"].fillna(mean_angle)

    # 4. Image Construction Helper
    def construct_images(df):
        images = []
        for i, row in df.iterrows():
            # Reshape flattened bands
            b1 = np.array(row["band_1"]).reshape(75, 75)
            b2 = np.array(row["band_2"]).reshape(75, 75)
            # 3rd Channel: Average
            avg = (b1 + b2) / 2.0

            # Stack: (3, 75, 75)
            img = np.stack([b1, b2, avg], axis=0)
            images.append(img)
        return np.array(images, dtype=np.float32)

    X_train_all = construct_images(df_train)
    X_test = construct_images(df_test)

    # 5. Normalization (Independent Per-Channel Min-Max Scaling)
    # Compute stats on the full labeled dataset
    # Shape: (N, 3, 75, 75) -> Min/Max over (0, 2, 3) -> (1, 3, 1, 1)
    min_val = X_train_all.min(axis=(0, 2, 3), keepdims=True)
    max_val = X_train_all.max(axis=(0, 2, 3), keepdims=True)

    # Avoid division by zero
    max_val[max_val == min_val] = min_val[max_val == min_val] + 1e-6

    X_train_all = (X_train_all - min_val) / (max_val - min_val)
    X_test = (X_test - min_val) / (max_val - min_val)

    # 6. Extract other arrays
    y_train_all = df_train["is_iceberg"].values.astype(np.float32)
    inc_train_all = df_train["inc_angle"].values.astype(np.float32)
    ids_train_all = df_train["id"].values

    inc_test = df_test["inc_angle"].values.astype(np.float32)
    ids_test = df_test["id"].values

    # 7. Load Metadata Splits (for fixed split usage)
    df_meta_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_meta_val = pd.read_csv(Config.VAL_META_PATH)

    fixed_train_ids = df_meta_train["id"].values
    fixed_val_ids = df_meta_val["id"].values

    # 8. Cache Data
    data_dict = {
        "X_train_all": X_train_all,
        "y_train_all": y_train_all,
        "inc_train_all": inc_train_all,
        "ids_train_all": ids_train_all,
        "X_test": X_test,
        "inc_test": inc_test,
        "ids_test": ids_test,
        "fixed_train_ids": fixed_train_ids,
        "fixed_val_ids": fixed_val_ids,
    }

    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
    np.savez_compressed(Config.PROCESSED_DATA_PATH, **data_dict)
    print(f"Data processed and saved to {Config.PROCESSED_DATA_PATH}")

    return data_dict


def get_data_loaders(fold=None, load_cached_data=True):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        fold (int, optional): If provided (0-4), performs Stratified K-Fold splitting.
                              If None, uses the fixed split from metadata files.
        load_cached_data (bool): Whether to use cached .npz data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load Data
    data = process_data(load_cached_data=load_cached_data)

    X_all = data["X_train_all"]
    y_all = data["y_train_all"]
    inc_all = data["inc_train_all"]
    ids_all = data["ids_train_all"]

    X_test = data["X_test"]
    inc_test = data["inc_test"]
    # ids_test = data['ids_test'] # Not needed for loader

    # Determine Train/Val Split
    if fold is not None:
        # Strategy: Stratified K-Fold on all labeled data
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # We only need the indices for the specific fold
        fold_generator = skf.split(X_all, y_all)
        for i, (train_idx, val_idx) in enumerate(fold_generator):
            if i == fold:
                break
        else:
            raise ValueError(f"Fold {fold} out of range (0-{Config.N_FOLDS-1})")

    else:
        # Strategy: Fixed Split from Metadata
        fixed_train_ids = set(data["fixed_train_ids"])
        fixed_val_ids = set(data["fixed_val_ids"])

        # Map IDs to indices
        # We assume IDs in json are unique
        id_to_idx = {id_: i for i, id_ in enumerate(ids_all)}

        train_idx = [id_to_idx[id_] for id_ in fixed_train_ids if id_ in id_to_idx]
        val_idx = [id_to_idx[id_] for id_ in fixed_val_ids if id_ in id_to_idx]

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

    # Create Datasets
    train_dataset = IcebergDataset(
        X_all[train_idx],
        inc_all[train_idx],
        y_all[train_idx],
        transform=True,  # Apply augmentation for training
    )

    val_dataset = IcebergDataset(
        X_all[val_idx], inc_all[val_idx], y_all[val_idx], transform=False
    )

    test_dataset = IcebergDataset(X_test, inc_test, labels=None, transform=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
