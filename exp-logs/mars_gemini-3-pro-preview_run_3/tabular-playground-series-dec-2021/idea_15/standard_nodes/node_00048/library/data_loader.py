import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import get_logger

logger = get_logger()


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type dataset.
    Handles both labeled (train/val) and unlabeled (test) data.
    """

    def __init__(self, X, y=None, ids=None, is_test=False):
        self.X = torch.FloatTensor(X)
        self.is_test = is_test

        if self.is_test:
            self.ids = ids
            self.y = None
        else:
            # Targets are 1-7, convert to 0-6 for PyTorch CrossEntropyLoss
            self.y = torch.LongTensor(y) - 1
            self.ids = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_test:
            return self.X[idx], self.ids[idx]
        else:
            return self.X[idx], self.y[idx]


def engineer_features(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    # 1. Cyclical Augmentation (Keep raw Aspect)
    # Aspect is in degrees (0-360)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude: Euclidean Distance to Hydrology
    # sqrt(Horizontal^2 + Vertical^2)
    df["Euclidean_Dist_Hydro"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation: Absolute Hydrology Elevation
    # Elevation - Vertical_Distance_To_Hydrology
    df["Abs_Hydro_Elev"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context: Mean Distance to Amenities
    # Mean of distances to Hydrology, Roadways, Fire Points
    df["Mean_Dist_Amenities"] = df[
        [
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    ].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads data, performs feature engineering and scaling, and caches the result.
    """
    # Define cache paths from Config
    cache_paths = {
        "train_X": Config.CACHE_TRAIN_X,
        "train_y": Config.CACHE_TRAIN_Y,
        "val_X": Config.CACHE_VAL_X,
        "val_y": Config.CACHE_VAL_Y,
        "test_X": Config.CACHE_TEST_X,
        "test_ids": Config.CACHE_TEST_IDS,
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and all_cached:
        logger.info("Loading cached data from working directory...")
        train_X = np.load(cache_paths["train_X"])
        train_y = np.load(cache_paths["train_y"])
        val_X = np.load(cache_paths["val_X"])
        val_y = np.load(cache_paths["val_y"])
        test_X = np.load(cache_paths["test_X"])
        test_ids = np.load(cache_paths["test_ids"])
        return train_X, train_y, val_X, val_y, test_X, test_ids

    logger.info("Processing data from scratch...")

    # Load Parquet files
    logger.info(f"Loading {Config.TRAIN_PATH}...")
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    logger.info(f"Loading {Config.VAL_PATH}...")
    df_val = pd.read_parquet(Config.VAL_PATH)
    logger.info(f"Loading {Config.TEST_PATH}...")
    df_test = pd.read_parquet(Config.TEST_PATH)

    # Separate Targets and IDs
    train_y = df_train["Cover_Type"].values
    val_y = df_val["Cover_Type"].values
    test_ids = df_test["Id"].values

    # Drop Id and Target from features
    # Note: Test set doesn't have Cover_Type
    drop_cols_train = ["Id", "Cover_Type"]
    drop_cols_test = ["Id"]

    df_train = df_train.drop(columns=drop_cols_train)
    df_val = df_val.drop(columns=drop_cols_train)
    df_test = df_test.drop(columns=drop_cols_test)

    # Apply Feature Engineering
    logger.info("Applying Augmented Physics-Informed Engineering...")
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Identify Continuous vs Binary Features
    # Binary features start with "Wilderness_Area" or "Soil_Type"
    # All others are continuous
    all_cols = df_train.columns.tolist()
    binary_cols = [
        c
        for c in all_cols
        if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
    ]
    continuous_cols = [c for c in all_cols if c not in binary_cols]

    logger.info(
        f"Identified {len(continuous_cols)} continuous features and {len(binary_cols)} binary features."
    )

    # Standardization
    # Fit only on Train, Transform all
    logger.info("Standardizing continuous features...")
    scaler = StandardScaler()

    # We process numpy arrays to save memory and ensure correct types
    train_cont = scaler.fit_transform(
        df_train[continuous_cols].values.astype(np.float32)
    )
    val_cont = scaler.transform(df_val[continuous_cols].values.astype(np.float32))
    test_cont = scaler.transform(df_test[continuous_cols].values.astype(np.float32))

    # Get binary parts
    train_bin = df_train[binary_cols].values.astype(np.float32)
    val_bin = df_val[binary_cols].values.astype(np.float32)
    test_bin = df_test[binary_cols].values.astype(np.float32)

    # Concatenate
    train_X = np.hstack([train_cont, train_bin])
    val_X = np.hstack([val_cont, val_bin])
    test_X = np.hstack([test_cont, test_bin])

    # Cache results
    logger.info(f"Caching processed data to {Config.WORKING_DIR}...")
    Config.setup()  # Ensure directory exists

    np.save(cache_paths["train_X"], train_X)
    np.save(cache_paths["train_y"], train_y)
    np.save(cache_paths["val_X"], val_X)
    np.save(cache_paths["val_y"], val_y)
    np.save(cache_paths["test_X"], test_X)
    np.save(cache_paths["test_ids"], test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(
        load_cached_data=load_cached_data
    )

    logger.info(
        f"Train Shape: {train_X.shape}, Val Shape: {val_X.shape}, Test Shape: {test_X.shape}"
    )

    train_dataset = CoverTypeDataset(train_X, train_y, is_test=False)
    val_dataset = CoverTypeDataset(val_X, val_y, is_test=False)
    test_dataset = CoverTypeDataset(test_X, ids=test_ids, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
