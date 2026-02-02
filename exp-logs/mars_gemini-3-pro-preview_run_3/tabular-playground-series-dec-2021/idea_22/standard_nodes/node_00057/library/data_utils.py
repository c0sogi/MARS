import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


def engineer_features(df):
    """
    Applies augmented physics-informed feature engineering.

    Transformations:
    1. Cyclical Aspect: Sine and Cosine of Aspect.
    2. Geometric Magnitude: Euclidean distance to hydrology.
    3. Directional Preservation: Absolute hydrology elevation.
    4. Global Context: Mean distance to amenities (Hydro, Road, Fire).
    """
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect)
    # Aspect is in degrees
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    h_dist = df["Horizontal_Distance_To_Hydrology"]
    v_dist = df["Vertical_Distance_To_Hydrology"]
    df["Euclidean_Dist_Hydro"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Distance preserves the "height above water" logic better than abs()
    df["Abs_Hydro_Elev"] = df["Elevation"] - v_dist

    # 4. Global Context (Mean Distance to Amenities)
    # We take absolute values for distances to treat them as magnitudes
    d_hydro = df["Horizontal_Distance_To_Hydrology"].abs()
    d_road = df["Horizontal_Distance_To_Roadways"].abs()
    d_fire = df["Horizontal_Distance_To_Fire_Points"].abs()
    df["Mean_Dist_Amenities"] = (d_hydro + d_road + d_fire) / 3.0

    return df


class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_dataloaders(load_cached_data=True):
    """
    Loads, processes, and caches data.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        train_loader, val_loader, test_loader, input_dim, test_ids
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
    }

    # Attempt to load from cache
    data_loaded = False
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_files.values()):
            print(f"Loading cached data from {Config.CACHE_DIR}...")
            try:
                X_train = np.load(cache_files["train_X"])
                y_train = np.load(cache_files["train_y"])
                X_val = np.load(cache_files["val_X"])
                y_val = np.load(cache_files["val_y"])
                X_test = np.load(cache_files["test_X"])
                test_ids = np.load(cache_files["test_ids"])
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}")
                data_loaded = False

    # Process from scratch if cache not loaded
    if not data_loaded:
        print("Processing data from scratch...")

        # Load Parquet files from Metadata
        df_train = pd.read_parquet(Config.TRAIN_PATH)
        df_val = pd.read_parquet(Config.VAL_PATH)
        df_test = pd.read_parquet(Config.TEST_PATH)

        # Apply Feature Engineering
        df_train = engineer_features(df_train)
        df_val = engineer_features(df_val)
        df_test = engineer_features(df_test)

        # Identify Columns
        target_col = "Cover_Type"
        id_col = "Id"

        # Identify Binary Columns (Wilderness Areas and Soil Types)
        # We select columns starting with these prefixes
        all_cols = df_train.columns
        bin_cols = [
            c
            for c in all_cols
            if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
        ]

        # Identify Continuous Columns (Everything else except Id and Target)
        exclude_cols = {target_col, id_col} | set(bin_cols)
        cont_cols = [c for c in all_cols if c not in exclude_cols]

        # Extract Targets (Convert 1-7 to 0-6 for CrossEntropyLoss)
        y_train = df_train[target_col].values - 1
        y_val = df_val[target_col].values - 1

        # Extract IDs for submission
        test_ids = df_test[id_col].values

        # Extract Continuous Features
        X_train_cont = df_train[cont_cols].values.astype(np.float32)
        X_val_cont = df_val[cont_cols].values.astype(np.float32)
        X_test_cont = df_test[cont_cols].values.astype(np.float32)

        # Extract Binary Features
        X_train_bin = df_train[bin_cols].values.astype(np.float32)
        X_val_bin = df_val[bin_cols].values.astype(np.float32)
        X_test_bin = df_test[bin_cols].values.astype(np.float32)

        # Standardization
        # Fit only on Train, Transform Train/Val/Test
        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(X_train_cont)
        X_val_cont = scaler.transform(X_val_cont)
        X_test_cont = scaler.transform(X_test_cont)

        # Concatenate Continuous and Binary Features
        X_train = np.hstack([X_train_cont, X_train_bin])
        X_val = np.hstack([X_val_cont, X_val_bin])
        X_test = np.hstack([X_test_cont, X_test_bin])

        # Save to cache
        print(f"Saving processed data to {Config.CACHE_DIR}...")
        np.save(cache_files["train_X"], X_train)
        np.save(cache_files["train_y"], y_train)
        np.save(cache_files["val_X"], X_val)
        np.save(cache_files["val_y"], y_val)
        np.save(cache_files["test_X"], X_test)
        np.save(cache_files["test_ids"], test_ids)

    # Create Datasets
    train_ds = ForestDataset(X_train, y_train)
    val_ds = ForestDataset(X_val, y_val)
    test_ds = ForestDataset(X_test, None)

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

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, input_dim, test_ids
