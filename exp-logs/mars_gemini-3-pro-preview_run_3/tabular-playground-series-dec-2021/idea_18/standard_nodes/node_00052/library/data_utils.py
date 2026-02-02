import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target vector.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def engineer_features(df):
    """
    Applies physics-informed feature engineering.
    Augments the dataframe with new columns.
    """
    # 1. Cyclical Augmentation (Keep raw Aspect as well)
    # Convert degrees to radians for sin/cos
    aspect_rad = df["Aspect"] * np.pi / 180.0
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude: Euclidean Distance to Hydrology
    # sqrt(H^2 + V^2)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation: Absolute Hydrology Elevation
    # Elevation - Vertical_Distance = Hydro_Elevation
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context: Mean Distance to Amenities
    # Average of distances to Hydrology, Roadways, Fire Points
    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, debug_sample_size=None
):
    """
    Orchestrates data loading, engineering, scaling, and caching.
    Returns DataLoaders for train, val, test, and the input feature dimension.
    """

    # Define cache paths
    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_X = np.load(cache_files["train_X"])
        train_y = np.load(cache_files["train_y"])
        val_X = np.load(cache_files["val_X"])
        val_y = np.load(cache_files["val_y"])
        test_X = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
    else:
        print("Processing data from scratch...")

        # Load Raw Data
        df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
        df_val = pd.read_parquet(Config.VAL_DATA_PATH)
        df_test = pd.read_parquet(Config.TEST_DATA_PATH)

        # Debugging: Subsample if requested
        if debug_sample_size is not None:
            df_train = df_train.iloc[:debug_sample_size]
            df_val = df_val.iloc[:debug_sample_size]
            df_test = df_test.iloc[:debug_sample_size]

        # Extract IDs for test set
        test_ids = df_test[Config.ID_COL].values

        # Feature Engineering
        print("Engineering features...")
        df_train = engineer_features(df_train)
        df_val = engineer_features(df_val)
        df_test = engineer_features(df_test)

        # Define Feature Groups
        # Original Continuous + New Engineered Continuous
        new_continuous = [
            "Aspect_Sin",
            "Aspect_Cos",
            "Euclidean_Distance_To_Hydrology",
            "Hydrology_Elevation",
            "Mean_Distance_To_Amenities",
        ]
        continuous_cols = Config.CONTINUOUS_FEATURES + new_continuous
        binary_cols = Config.BINARY_FEATURES

        # Extract Targets (Map 1-7 to 0-6)
        train_y = df_train[Config.TARGET_COL].values - 1
        val_y = df_val[Config.TARGET_COL].values - 1

        # Prepare Features
        # 1. Continuous Features (Standardized)
        scaler = StandardScaler()

        # Fit only on Train
        train_cont = scaler.fit_transform(
            df_train[continuous_cols].values.astype(np.float32)
        )
        val_cont = scaler.transform(df_val[continuous_cols].values.astype(np.float32))
        test_cont = scaler.transform(df_test[continuous_cols].values.astype(np.float32))

        # 2. Binary Features (Raw)
        train_bin = df_train[binary_cols].values.astype(np.float32)
        val_bin = df_val[binary_cols].values.astype(np.float32)
        test_bin = df_test[binary_cols].values.astype(np.float32)

        # Concatenate
        train_X = np.hstack([train_cont, train_bin])
        val_X = np.hstack([val_cont, val_bin])
        test_X = np.hstack([test_cont, test_bin])

        # Save to Cache
        print(f"Saving processed data to {Config.CACHE_DIR}...")
        np.save(cache_files["train_X"], train_X)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["val_X"], val_X)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["test_X"], test_X)
        np.save(cache_files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = ForestDataset(train_X, train_y)
    val_dataset = ForestDataset(val_X, val_y)
    test_dataset = ForestDataset(test_X, y=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    input_dim = train_X.shape[1]

    return train_loader, val_loader, test_loader, test_ids, input_dim
