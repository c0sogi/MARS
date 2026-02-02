import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering to the dataframe.
    """
    # 1. Cyclical Augmentation (Keep raw Aspect)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context
    amenities = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Distance_To_Amenities"] = df[amenities].mean(axis=1)

    return df


class CoverTypeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            # Map class labels 1-7 to 0-6 for CrossEntropyLoss
            self.y = torch.tensor(y - 1, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    Returns: X_train, y_train, X_val, y_val, X_test, test_ids
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(Config.WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(Config.WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(Config.WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(Config.WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
    }

    # Try to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_files.values()):
            return (
                np.load(cache_files["X_train"]),
                np.load(cache_files["y_train"]),
                np.load(cache_files["X_val"]),
                np.load(cache_files["y_val"]),
                np.load(cache_files["X_test"]),
                np.load(cache_files["test_ids"]),
            )

    # Process from scratch
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    # Apply Feature Engineering
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    target_col = "Cover_Type"
    id_col = "Id"

    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    test_ids = df_test[id_col].values

    drop_cols = [target_col, id_col]
    X_train_df = df_train.drop(columns=drop_cols, errors="ignore")
    X_val_df = df_val.drop(columns=drop_cols, errors="ignore")
    X_test_df = df_test.drop(columns=[id_col], errors="ignore")

    # Separate Continuous and Binary Features
    cols = X_train_df.columns
    binary_cols = [
        c for c in cols if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]
    continuous_cols = [c for c in cols if c not in binary_cols]

    # Standardize Continuous Features (Fit on Train, Transform Val/Test)
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(
        X_train_df[continuous_cols].values.astype(np.float32)
    )
    X_val_cont = scaler.transform(X_val_df[continuous_cols].values.astype(np.float32))
    X_test_cont = scaler.transform(X_test_df[continuous_cols].values.astype(np.float32))

    # Keep Binary Features as is
    X_train_bin = X_train_df[binary_cols].values.astype(np.float32)
    X_val_bin = X_val_df[binary_cols].values.astype(np.float32)
    X_test_bin = X_test_df[binary_cols].values.astype(np.float32)

    # Concatenate
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True, num_workers=4):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    X_train, y_train, X_val, y_val, X_test, _ = process_data(load_cached_data)

    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)
    test_dataset = CoverTypeDataset(X_test, None)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
