import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import (
    METADATA_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    TRAIN_CACHE_PATH,
    TRAIN_LABELS_PATH,
    VAL_CACHE_PATH,
    VAL_LABELS_PATH,
    TEST_CACHE_PATH,
    TEST_IDS_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything


def engineer_features(df):
    """
    Applies physics-informed feature augmentation based on the strategy.
    """
    # 1. Cyclical Aspect (Preserve raw Aspect as well)
    # Convert degrees to radians for sin/cos
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180.0)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180.0)

    # 2. Euclidean Distance to Hydrology (Hypotenuse)
    # sqrt(Horizontal^2 + Vertical^2)
    df["Hydrology_Distance_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Absolute Hydrology Elevation (Directional preservation)
    # Vertical_Dist = Elev - Hydro_Elev -> Hydro_Elev = Elev - Vertical_Dist
    df["Hydrology_Elevation_Abs"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Amenities Mean Distance (Global context shortcut)
    df["Amenities_Mean_Dist"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


class CoverTypeDataset(Dataset):
    """
    Custom Dataset for Cover Type prediction.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Loads data, performs feature engineering and scaling, and returns DataLoaders.
    Implements caching to avoid re-processing.
    """
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = [
        TRAIN_CACHE_PATH,
        TRAIN_LABELS_PATH,
        VAL_CACHE_PATH,
        VAL_LABELS_PATH,
        TEST_CACHE_PATH,
        TEST_IDS_PATH,
    ]

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        train_X = np.load(TRAIN_CACHE_PATH)
        train_y = np.load(TRAIN_LABELS_PATH)
        val_X = np.load(VAL_CACHE_PATH)
        val_y = np.load(VAL_LABELS_PATH)
        test_X = np.load(TEST_CACHE_PATH)
        test_ids = np.load(TEST_IDS_PATH)
    else:
        print("Processing data from scratch...")

        # Load Parquet Metadata
        df_train = pd.read_parquet(TRAIN_DATA_PATH)
        df_val = pd.read_parquet(VAL_DATA_PATH)
        df_test = pd.read_parquet(TEST_DATA_PATH)

        # Extract Targets and IDs
        train_y = df_train["Cover_Type"].values
        val_y = df_val["Cover_Type"].values
        test_ids = df_test["Id"].values

        # Drop Id and Target from feature sets
        # Train/Val have Cover_Type, Test does not
        X_train_df = df_train.drop(columns=["Id", "Cover_Type"])
        X_val_df = df_val.drop(columns=["Id", "Cover_Type"])
        X_test_df = df_test.drop(columns=["Id"])

        # Feature Engineering
        X_train_df = engineer_features(X_train_df)
        X_val_df = engineer_features(X_val_df)
        X_test_df = engineer_features(X_test_df)

        # Identify Columns
        # Binary columns: Soil_Type* and Wilderness_Area*
        binary_cols = [
            c
            for c in X_train_df.columns
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]
        # Continuous columns: All others
        continuous_cols = [c for c in X_train_df.columns if c not in binary_cols]

        # Standardization (Continuous only)
        scaler = StandardScaler()

        # Fit on Train, Transform all
        X_train_df[continuous_cols] = scaler.fit_transform(
            X_train_df[continuous_cols].astype(np.float32)
        )
        X_val_df[continuous_cols] = scaler.transform(
            X_val_df[continuous_cols].astype(np.float32)
        )
        X_test_df[continuous_cols] = scaler.transform(
            X_test_df[continuous_cols].astype(np.float32)
        )

        # Convert to Numpy (float32) and ensure column order
        all_cols = continuous_cols + binary_cols

        train_X = X_train_df[all_cols].values.astype(np.float32)
        val_X = X_val_df[all_cols].values.astype(np.float32)
        test_X = X_test_df[all_cols].values.astype(np.float32)

        # Adjust Targets to 0-indexed (1-7 -> 0-6)
        train_y = (train_y - 1).astype(np.int64)
        val_y = (val_y - 1).astype(np.int64)

        # Save to Cache
        np.save(TRAIN_CACHE_PATH, train_X)
        np.save(TRAIN_LABELS_PATH, train_y)
        np.save(VAL_CACHE_PATH, val_X)
        np.save(VAL_LABELS_PATH, val_y)
        np.save(TEST_CACHE_PATH, test_X)
        np.save(TEST_IDS_PATH, test_ids)

    # Instantiate Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X, y=None)

    # Create DataLoaders
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

    input_dim = train_X.shape[1]

    return train_loader, val_loader, test_loader, input_dim, test_ids
