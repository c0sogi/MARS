import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    FINAL_CONTINUOUS_FEATURES,
    FINAL_BINARY_FEATURES,
    LABEL_TO_IDX,
    TARGET_COL,
    ID_COL,
    RAW_CONTINUOUS_FEATURES,
    FEAT_ASPECT_SIN,
    FEAT_ASPECT_COS,
    FEAT_EUCLIDEAN_HYDRO,
    FEAT_ABS_HYDRO_ELEV,
    FEAT_MEAN_AMENITIES,
)


def feature_engineering(df):
    """
    Applies physics-informed feature engineering to the dataframe.
    """
    # Ensure we don't modify the original dataframe in place unexpectedly
    df = df.copy()

    # 1. Cyclical Augmentation for Aspect
    # Aspect is in degrees (0-360). Convert to radians.
    # We retain the raw 'Aspect' column as per strategy.
    if "Aspect" in df.columns:
        df[FEAT_ASPECT_SIN] = np.sin(df["Aspect"] * np.pi / 180.0)
        df[FEAT_ASPECT_COS] = np.cos(df["Aspect"] * np.pi / 180.0)

    # 2. Geometric Magnitude: Euclidean Distance to Hydrology
    # sqrt(Horizontal^2 + Vertical^2)
    if (
        "Horizontal_Distance_To_Hydrology" in df.columns
        and "Vertical_Distance_To_Hydrology" in df.columns
    ):
        df[FEAT_EUCLIDEAN_HYDRO] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

    # 3. Directional Preservation: Absolute Hydrology Elevation
    # Elevation - Vertical_Distance_To_Hydrology
    if "Elevation" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
        df[FEAT_ABS_HYDRO_ELEV] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context: Mean Distance to Amenities
    # Average of distances to Hydrology, Roadways, and Fire Points
    amenity_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    if all(col in df.columns for col in amenity_cols):
        df[FEAT_MEAN_AMENITIES] = df[amenity_cols].mean(axis=1)

    return df


class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def process_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and caching.
    Returns processed numpy arrays: train_X, train_y, val_X, val_y, test_X, test_ids
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    cache_files = {
        "train_X": os.path.join(CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading cached data from", CACHE_DIR)
        train_X = np.load(cache_files["train_X"])
        train_y = np.load(cache_files["train_y"])
        val_X = np.load(cache_files["val_X"])
        val_y = np.load(cache_files["val_y"])
        test_X = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
        return train_X, train_y, val_X, val_y, test_X, test_ids

    print("Processing data from scratch...")

    # Load metadata parquets
    df_train = pd.read_parquet(TRAIN_DATA_PATH)
    df_val = pd.read_parquet(VAL_DATA_PATH)
    df_test = pd.read_parquet(TEST_DATA_PATH)

    # Extract Test IDs before processing
    test_ids = df_test[ID_COL].values

    # Apply Feature Engineering
    print("Applying feature engineering...")
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Prepare Feature Subsets
    # Continuous features to be standardized
    cont_cols = FINAL_CONTINUOUS_FEATURES
    # Binary features to be kept as is (0/1)
    bin_cols = FINAL_BINARY_FEATURES

    # Fit StandardScaler on Training Continuous Features ONLY
    print("Fitting scaler...")
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
    val_cont = scaler.transform(df_val[cont_cols].values.astype(np.float32))
    test_cont = scaler.transform(df_test[cont_cols].values.astype(np.float32))

    # Extract Binary Features
    train_bin = df_train[bin_cols].values.astype(np.float32)
    val_bin = df_val[bin_cols].values.astype(np.float32)
    test_bin = df_test[bin_cols].values.astype(np.float32)

    # Concatenate Features
    train_X = np.hstack([train_cont, train_bin])
    val_X = np.hstack([val_cont, val_bin])
    test_X = np.hstack([test_cont, test_bin])

    # Process Targets
    # Map class labels to 0-N indices
    print("Processing targets...")
    train_y_raw = df_train[TARGET_COL].values
    val_y_raw = df_val[TARGET_COL].values

    # Vectorized mapping
    train_y = np.zeros_like(train_y_raw)
    val_y = np.zeros_like(val_y_raw)

    for label, idx in LABEL_TO_IDX.items():
        train_y[train_y_raw == label] = idx
        val_y[val_y_raw == label] = idx

    # Save to cache
    print("Saving data to cache...")
    np.save(cache_files["train_X"], train_X)
    np.save(cache_files["train_y"], train_y)
    np.save(cache_files["val_X"], val_X)
    np.save(cache_files["val_y"], val_y)
    np.save(cache_files["test_X"], test_X)
    np.save(cache_files["test_ids"], test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(load_cached_data)

    train_dataset = ForestDataset(train_X, train_y)
    val_dataset = ForestDataset(val_X, val_y)
    test_dataset = ForestDataset(test_X, y=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, test_ids
