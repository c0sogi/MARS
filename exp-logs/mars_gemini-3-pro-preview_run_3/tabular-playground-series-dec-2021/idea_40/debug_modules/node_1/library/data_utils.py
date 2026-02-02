import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_col_name(df, prefix):
    """Helper to find column name by prefix to handle potential naming variations."""
    for col in df.columns:
        if col.startswith(prefix):
            return col
    # Fallback or raise error if critical
    raise ValueError(f"Column starting with '{prefix}' not found in dataframe.")


def feature_engineering(df):
    """
    Applies physics-informed feature engineering.
    """
    df = df.copy()

    # Identify dynamic column names
    h_hydro = get_col_name(df, "Horizontal_Distance_To_Hydrol")
    v_hydro = get_col_name(df, "Vertical_Distance_To_Hydrolog")
    h_road = get_col_name(df, "Horizontal_Distance_To_Roadwa")
    h_fire = get_col_name(df, "Horizontal_Distance_To_Fire_P")
    elevation = "Elevation"
    aspect = "Aspect"

    # 1. Cyclical Augmentation
    # Convert degrees to radians for sin/cos
    df["Aspect_Sin"] = np.sin(np.radians(df[aspect]))
    df["Aspect_Cos"] = np.cos(np.radians(df[aspect]))

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(h^2 + v^2)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(df[h_hydro] ** 2 + df[v_hydro] ** 2)

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Distance_To_Hydrology
    df["Absolute_Hydrology_Elevation"] = df[elevation] - df[v_hydro]

    # 4. Global Context (Mean Distance to Amenities)
    # Average of Horizontal distances to Hydro, Road, Fire
    df["Mean_Distance_To_Amenities"] = (df[h_hydro] + df[h_road] + df[h_fire]) / 3.0

    return df


def get_dataloaders(load_cached_data=True):
    """
    Loads data, performs feature engineering/scaling, and returns DataLoaders.
    Implements caching mechanism.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(Config.WORKING_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.WORKING_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.WORKING_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.WORKING_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.WORKING_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        X_train = np.load(cache_files["train_X"])
        y_train = np.load(cache_files["train_y"])
        X_val = np.load(cache_files["val_X"])
        y_val = np.load(cache_files["val_y"])
        X_test = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])

    else:
        print("Processing data from scratch...")

        # Load metadata parquets
        print(f"Loading {Config.TRAIN_PATH}...")
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        print(f"Loading {Config.VAL_PATH}...")
        val_df = pd.read_parquet(Config.VAL_PATH)
        print(f"Loading {Config.TEST_PATH}...")
        test_df = pd.read_parquet(Config.TEST_PATH)

        # Extract Targets and IDs
        # Shift targets to 0-indexed (1-7 -> 0-6)
        y_train = train_df[Config.TARGET_COL].values.astype(np.int64) - 1
        y_val = val_df[Config.TARGET_COL].values.astype(np.int64) - 1
        test_ids = test_df[Config.ID_COL].values.astype(np.int64)

        # Drop ID and Target columns to isolate features
        drop_cols_train = [Config.ID_COL, Config.TARGET_COL]
        drop_cols_test = [Config.ID_COL]

        X_train_df = train_df.drop(columns=drop_cols_train, errors="ignore")
        X_val_df = val_df.drop(columns=drop_cols_train, errors="ignore")
        X_test_df = test_df.drop(columns=drop_cols_test, errors="ignore")

        # Apply Feature Engineering
        print("Applying feature engineering...")
        X_train_df = feature_engineering(X_train_df)
        X_val_df = feature_engineering(X_val_df)
        X_test_df = feature_engineering(X_test_df)

        # Separate Continuous and Binary Features
        # Binary features: Soil_Type*, Wilderness_Area*
        all_cols = X_train_df.columns.tolist()
        binary_cols = [
            c
            for c in all_cols
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]
        continuous_cols = [c for c in all_cols if c not in binary_cols]

        print(f"Continuous features: {len(continuous_cols)}")
        print(f"Binary features: {len(binary_cols)}")

        # Standard Scaling for Continuous Features
        # Fit ONLY on Train
        print("Scaling continuous features...")
        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(X_train_df[continuous_cols])
        X_val_cont = scaler.transform(X_val_df[continuous_cols])
        X_test_cont = scaler.transform(X_test_df[continuous_cols])

        # Retrieve Binary Features (already 0/1)
        X_train_bin = X_train_df[binary_cols].values
        X_val_bin = X_val_df[binary_cols].values
        X_test_bin = X_test_df[binary_cols].values

        # Concatenate Features
        # Ensure float32 for model compatibility
        X_train = np.hstack([X_train_cont, X_train_bin]).astype(np.float32)
        X_val = np.hstack([X_val_cont, X_val_bin]).astype(np.float32)
        X_test = np.hstack([X_test_cont, X_test_bin]).astype(np.float32)

        # Save to Cache
        print(f"Saving processed data to {Config.WORKING_DIR}...")
        np.save(cache_files["train_X"], X_train)
        np.save(cache_files["train_y"], y_train)
        np.save(cache_files["val_X"], X_val)
        np.save(cache_files["val_y"], y_val)
        np.save(cache_files["test_X"], X_test)
        np.save(cache_files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = ForestDataset(X_train, y_train)
    val_dataset = ForestDataset(X_val, y_val)
    test_dataset = ForestDataset(X_test, None)

    # Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, test_ids
