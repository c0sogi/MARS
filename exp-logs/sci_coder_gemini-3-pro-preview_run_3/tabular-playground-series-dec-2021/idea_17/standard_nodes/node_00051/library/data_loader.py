import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class ForestCoverDataset(Dataset):
    def __init__(self, X, y=None):
        """
        PyTorch Dataset for Forest Cover Type.

        Args:
            X (np.ndarray): Feature matrix (float32).
            y (np.ndarray, optional): Target labels (int64).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def engineer_features(df):
    """
    Applies the Augmented Physics-Informed Engineering strategy.

    1. Cyclical Augmentation: Aspect_Sin/Cos (retaining raw Aspect).
    2. Geometric Magnitude: Euclidean Distance to Hydrology.
    3. Directional Preservation: Absolute Hydrology Elevation.
    4. Global Context: Mean Distance to Amenities.
    """
    # Avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation
    # Aspect is in degrees [0, 360]
    # We retain the raw Aspect column as per instructions (it's already in df)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude
    # Euclidean Distance to Hydrology = sqrt(H^2 + V^2)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation
    # Absolute Hydrology Elevation = Elevation - Vertical_Distance_To_Hydrolog
    # (Since Vertical_Dist = Elev - Hydro_Elev)
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context
    # Mean Distance to Amenities (Hydrology, Roadways, Fire Points)
    amenities = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Distance_To_Amenities"] = df[amenities].mean(axis=1)

    return df


def get_dataloaders(load_cached_data=True):
    """
    Loads data, performs feature engineering/preprocessing, and returns DataLoaders.
    Implements caching mechanism strictly as requested.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
                                 If False or load fails, re-processes and saves.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    seed_everything()

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "train_X": os.path.join(Config.CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_paths.values())
        if all_exist:
            print(f"Loading cached data from {Config.CACHE_DIR}...")
            try:
                train_X = np.load(cache_paths["train_X"])
                train_y = np.load(cache_paths["train_y"])
                val_X = np.load(cache_paths["val_X"])
                val_y = np.load(cache_paths["val_y"])
                test_X = np.load(cache_paths["test_X"])
                test_ids = np.load(cache_paths["test_ids"])

                # Create DataLoaders
                train_ds = ForestCoverDataset(train_X, train_y)
                val_ds = ForestCoverDataset(val_X, val_y)
                test_ds = ForestCoverDataset(test_X, None)

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

                return train_loader, val_loader, test_loader, test_ids
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")
        else:
            print("Cache missing. Reprocessing...")
    else:
        print("Force reprocessing (load_cached_data=False)...")

    # 2. Process from scratch
    print("Loading parquet files...")
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    # Debug Subsampling
    if Config.DEBUG:
        print(f"DEBUG Mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Extract Targets and IDs
    # Map labels: 1->0, 2->1, etc.
    train_y = df_train["Cover_Type"].map(Config.LABEL_MAP).values.astype(np.int64)
    val_y = df_val["Cover_Type"].map(Config.LABEL_MAP).values.astype(np.int64)
    test_ids = df_test["Id"].values

    # Drop non-feature columns
    drop_cols_train = ["Id", "Cover_Type"]
    drop_cols_test = ["Id"]

    df_train = df_train.drop(columns=drop_cols_train, errors="ignore")
    df_val = df_val.drop(columns=drop_cols_train, errors="ignore")
    df_test = df_test.drop(columns=drop_cols_test, errors="ignore")

    # Feature Engineering
    print("Engineering features...")
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Identify Column Groups
    # Continuous features: Original + New
    new_continuous = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Distance_To_Hydrology",
        "Absolute_Hydrology_Elevation",
        "Mean_Distance_To_Amenities",
    ]
    continuous_cols = Config.CONTINUOUS_FEATURES + new_continuous

    # Binary features: All others (Soil Types, Wilderness Areas)
    all_cols = df_train.columns.tolist()
    binary_cols = [c for c in all_cols if c not in continuous_cols]

    # Standardization
    print("Standardizing continuous features...")
    scaler = StandardScaler()
    scaler.fit(df_train[continuous_cols])

    # Transform continuous parts
    train_cont = scaler.transform(df_train[continuous_cols])
    val_cont = scaler.transform(df_val[continuous_cols])
    test_cont = scaler.transform(df_test[continuous_cols])

    # Get binary parts (no scaling, keep as is)
    train_bin = df_train[binary_cols].values
    val_bin = df_val[binary_cols].values
    test_bin = df_test[binary_cols].values

    # Concatenate to form dense feature vector
    # Order: Continuous features first, then Binary features
    train_X = np.hstack([train_cont, train_bin]).astype(np.float32)
    val_X = np.hstack([val_cont, val_bin]).astype(np.float32)
    test_X = np.hstack([test_cont, test_bin]).astype(np.float32)

    # Save to cache
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    np.save(cache_paths["train_X"], train_X)
    np.save(cache_paths["train_y"], train_y)
    np.save(cache_paths["val_X"], val_X)
    np.save(cache_paths["val_y"], val_y)
    np.save(cache_paths["test_X"], test_X)
    np.save(cache_paths["test_ids"], test_ids)

    # Create DataLoaders
    train_ds = ForestCoverDataset(train_X, train_y)
    val_ds = ForestCoverDataset(val_X, val_y)
    test_ds = ForestCoverDataset(test_X, None)

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

    return train_loader, val_loader, test_loader, test_ids
