import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config

# Define original continuous columns
ORIG_CONTINUOUS_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]


def feature_engineering(df):
    """
    Applies physics-informed geometric feature engineering.
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # 1. Euclidean Distance to Hydrology
    # sqrt(H^2 + V^2)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 2. Composite Aggregates (Linear Combinations)
    # Sum of distances to key points
    df["Sum_Dist_Road_Fire"] = (
        df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Sum_Dist_Hydro_Road"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    )
    df["Sum_Dist_Hydro_Fire"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )

    # 3. Absolute Vertical Distance
    df["Abs_Vertical_Distance_To_Hydrology"] = df[
        "Vertical_Distance_To_Hydrology"
    ].abs()

    # 4. Mean Distance to Amenities (Cite Lesson 00009)
    df["Mean_Distance_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def get_continuous_columns():
    """Returns the list of continuous columns including engineered ones."""
    new_cols = [
        "Euclidean_Distance_To_Hydrology",
        "Sum_Dist_Road_Fire",
        "Sum_Dist_Hydro_Road",
        "Sum_Dist_Hydro_Fire",
        "Abs_Vertical_Distance_To_Hydrology",
        "Mean_Distance_Amenities",
    ]
    return ORIG_CONTINUOUS_COLS + new_cols


def preprocess_data(train_df, val_df, test_df):
    """
    Scales continuous features and prepares X/y arrays.
    """
    # 1. Feature Engineering
    print("Applying feature engineering...")
    train_df = feature_engineering(train_df)
    val_df = feature_engineering(val_df)
    test_df = feature_engineering(test_df)

    # 2. Separate Target and IDs
    # Train
    train_y = train_df[Config.TARGET_COL].values - 1  # 0-indexed
    train_ids = train_df[Config.ID_COL].values
    train_df = train_df.drop(columns=[Config.TARGET_COL, Config.ID_COL])

    # Val
    val_y = val_df[Config.TARGET_COL].values - 1  # 0-indexed
    val_ids = val_df[Config.ID_COL].values
    val_df = val_df.drop(columns=[Config.TARGET_COL, Config.ID_COL])

    # Test
    test_ids = test_df[Config.ID_COL].values
    if Config.TARGET_COL in test_df.columns:
        test_df = test_df.drop(columns=[Config.TARGET_COL])
    test_df = test_df.drop(columns=[Config.ID_COL])

    # 3. Identify Feature Groups
    continuous_cols = get_continuous_columns()
    # Binary columns are everything else
    binary_cols = [c for c in train_df.columns if c not in continuous_cols]

    print(f"Continuous features: {len(continuous_cols)}")
    print(f"Binary features: {len(binary_cols)}")

    # 4. Standardization
    # Fit scaler only on training data
    scaler = StandardScaler()

    train_cont = scaler.fit_transform(
        train_df[continuous_cols].values.astype(np.float32)
    )
    val_cont = scaler.transform(val_df[continuous_cols].values.astype(np.float32))
    test_cont = scaler.transform(test_df[continuous_cols].values.astype(np.float32))

    # Binary features (no scaling needed)
    train_bin = train_df[binary_cols].values.astype(np.float32)
    val_bin = val_df[binary_cols].values.astype(np.float32)
    test_bin = test_df[binary_cols].values.astype(np.float32)

    # 5. Concatenate
    train_X = np.hstack([train_cont, train_bin])
    val_X = np.hstack([val_cont, val_bin])
    test_X = np.hstack([test_cont, test_bin])

    return train_X, train_y, val_X, val_y, test_X, test_ids


def load_data(config=Config, load_cached_data=True):
    """
    Loads data, performing feature engineering and preprocessing.
    Uses caching to speed up subsequent runs.
    """
    Config.setup()  # Ensure directories exist

    # Check cache
    files_exist = (
        os.path.exists(config.TRAIN_X_PATH)
        and os.path.exists(config.TRAIN_Y_PATH)
        and os.path.exists(config.VAL_X_PATH)
        and os.path.exists(config.VAL_Y_PATH)
        and os.path.exists(config.TEST_X_PATH)
        and os.path.exists(config.TEST_IDS_PATH)
    )

    if load_cached_data and files_exist:
        print("Loading cached data from .npy files...")
        train_X = np.load(config.TRAIN_X_PATH)
        train_y = np.load(config.TRAIN_Y_PATH)
        val_X = np.load(config.VAL_X_PATH)
        val_y = np.load(config.VAL_Y_PATH)
        test_X = np.load(config.TEST_X_PATH)
        test_ids = np.load(config.TEST_IDS_PATH)
    else:
        print("Loading raw data from parquet...")
        train_df = pd.read_parquet(config.TRAIN_METADATA_PATH)
        val_df = pd.read_parquet(config.VAL_METADATA_PATH)
        test_df = pd.read_parquet(config.TEST_METADATA_PATH)

        if config.DEBUG:
            print(f"DEBUG: Subsampling {config.DEBUG_SUBSET_SIZE} rows...")
            train_df = train_df.iloc[: config.DEBUG_SUBSET_SIZE]
            val_df = val_df.iloc[: config.DEBUG_SUBSET_SIZE]
            test_df = test_df.iloc[: config.DEBUG_SUBSET_SIZE]

        print("Preprocessing and engineering features...")
        train_X, train_y, val_X, val_y, test_X, test_ids = preprocess_data(
            train_df, val_df, test_df
        )

        print("Saving processed data to cache...")
        np.save(config.TRAIN_X_PATH, train_X)
        np.save(config.TRAIN_Y_PATH, train_y)
        np.save(config.VAL_X_PATH, val_X)
        np.save(config.VAL_Y_PATH, val_y)
        np.save(config.TEST_X_PATH, test_X)
        np.save(config.TEST_IDS_PATH, test_ids)

    print(f"Train shape: {train_X.shape}")
    print(f"Val shape: {val_X.shape}")
    print(f"Test shape: {test_X.shape}")

    return train_X, train_y, val_X, val_y, test_X, test_ids


def get_dataloaders(config=Config, load_cached_data=True):
    """
    Returns PyTorch DataLoaders for train, val, and test sets.
    """
    train_X, train_y, val_X, val_y, test_X, test_ids = load_data(
        config, load_cached_data
    )

    # Convert to Tensors
    # Features are float32, Targets are long (int64)
    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long)
    )
    # Test dataset only has features
    test_dataset = TensorDataset(torch.tensor(test_X, dtype=torch.float32))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
