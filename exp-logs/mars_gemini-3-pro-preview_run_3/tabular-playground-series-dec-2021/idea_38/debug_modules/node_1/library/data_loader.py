import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    TARGET_COL,
    ID_COL,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
)


def engineer_features(df):
    """
    Applies physics-informed feature engineering to the dataframe.
    Adds cyclical aspect, geometric hydrology distance, absolute hydrology elevation,
    and mean amenity distance.
    """
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect as well)
    # Convert Aspect (degrees) to radians for sin/cos
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude: Euclidean Distance to Hydrology
    # sqrt(Horizontal^2 + Vertical^2)
    df["Euclidean_Dist_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation: Absolute Hydrology Elevation
    # Elevation - Vertical_Distance (preserves flow directionality)
    df["Abs_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context: Mean Distance to Amenities
    # Mean of distances to Hydrology, Roadways, and Fire Points
    df["Mean_Dist_Amenities"] = df[
        [
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    ].mean(axis=1)

    return df


def get_feature_groups(df):
    """
    Identifies continuous and binary feature columns.
    """
    exclude = [ID_COL, TARGET_COL]
    all_cols = [c for c in df.columns if c not in exclude]

    # Binary features are Soil_Type* and Wilderness_Area*
    binary_cols = [
        c
        for c in all_cols
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]

    # Continuous features are everything else
    continuous_cols = [c for c in all_cols if c not in binary_cols]

    return continuous_cols, binary_cols


def preprocess_data(load_cached_data=True):
    """
    Loads data, performs feature engineering and scaling, and handles caching.
    Returns processed numpy arrays for train, val, and test sets.
    """
    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Check if we can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print(f"Loading cached data from {CACHE_DIR}...")
            data = {k: np.load(v) for k, v in cache_files.items()}
            return (
                data["train_X"],
                data["train_y"],
                data["val_X"],
                data["val_y"],
                data["test_X"],
                data["test_ids"],
            )
        else:
            print("Cache incomplete or missing. Processing data from scratch...")

    # Load raw data from metadata parquet files
    print("Loading metadata...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Debugging: Subsample if requested
    if MAX_TRAIN_SAMPLES is not None:
        print(f"Subsampling training data to {MAX_TRAIN_SAMPLES} rows.")
        train_df = train_df.iloc[:MAX_TRAIN_SAMPLES]
    if MAX_VAL_SAMPLES is not None:
        print(f"Subsampling validation data to {MAX_VAL_SAMPLES} rows.")
        val_df = val_df.iloc[:MAX_VAL_SAMPLES]

    # Feature Engineering
    print("Applying feature engineering...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Identify feature groups
    cont_cols, bin_cols = get_feature_groups(train_df)

    # Preprocessing
    print("Scaling continuous features...")
    scaler = StandardScaler()

    # Fit scaler on training continuous features only
    train_cont = scaler.fit_transform(train_df[cont_cols].values.astype(np.float32))
    val_cont = scaler.transform(val_df[cont_cols].values.astype(np.float32))
    test_cont = scaler.transform(test_df[cont_cols].values.astype(np.float32))

    # Binary features remain raw (0/1)
    train_bin = train_df[bin_cols].values.astype(np.float32)
    val_bin = val_df[bin_cols].values.astype(np.float32)
    test_bin = test_df[bin_cols].values.astype(np.float32)

    # Concatenate features
    train_X = np.hstack([train_cont, train_bin])
    val_X = np.hstack([val_cont, val_bin])
    test_X = np.hstack([test_cont, test_bin])

    # Process Targets: Shift from 1-7 to 0-6
    train_y = (train_df[TARGET_COL].values - 1).astype(np.int64)
    val_y = (val_df[TARGET_COL].values - 1).astype(np.int64)

    # Extract Test IDs
    test_ids = test_df[ID_COL].values

    # Save to cache
    print(f"Saving processed data to {CACHE_DIR}...")
    np.save(cache_files["train_X"], train_X)
    np.save(cache_files["train_y"], train_y)
    np.save(cache_files["val_X"], val_X)
    np.save(cache_files["val_y"], val_y)
    np.save(cache_files["test_X"], test_X)
    np.save(cache_files["test_ids"], test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


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
    Orchestrates data loading and returns PyTorch DataLoaders.
    Returns: train_loader, val_loader, test_loader, test_ids, input_dim
    """
    train_X, train_y, val_X, val_y, test_X, test_ids = preprocess_data(load_cached_data)

    # Create Datasets
    train_dataset = ForestDataset(train_X, train_y)
    val_dataset = ForestDataset(val_X, val_y)
    test_dataset = ForestDataset(test_X)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    input_dim = train_X.shape[1]

    return train_loader, val_loader, test_loader, test_ids, input_dim
