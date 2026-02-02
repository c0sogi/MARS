import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    Adds cyclical aspect, geometric magnitudes, and global context features
    while preserving raw columns.
    """
    # Create a copy to avoid SettingWithCopy warnings on the original dataframe
    df = df.copy()

    # 1. Cyclical Augmentation
    # Convert degrees to radians
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180)

    # 2. Geometric Magnitude
    # Euclidean Distance to Hydrology (Hypotenuse)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation
    # Absolute Hydrology Elevation (preserves uphill/downhill context relative to water)
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context
    # Mean Distance to Amenities (Hydrology, Roadways, Fire Points)
    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def preprocess_data(df_train, df_val, df_test):
    """
    Standardizes continuous features, prepares binary features, and encodes targets.
    """
    # Extract Targets and IDs
    y_train_raw = df_train[Config.TARGET_COL].values
    y_val_raw = df_val[Config.TARGET_COL].values
    test_ids = df_test[Config.ID_COL].values

    # Drop non-feature columns (ID and Target)
    drop_cols = [Config.ID_COL, Config.TARGET_COL]
    X_train_df = df_train.drop(columns=drop_cols, errors="ignore")
    X_val_df = df_val.drop(columns=drop_cols, errors="ignore")
    X_test_df = df_test.drop(columns=[Config.ID_COL], errors="ignore")

    # Define Feature Groups
    # Base continuous columns from the dataset
    base_cont_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
    ]
    # Newly engineered continuous columns
    new_cont_cols = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Euclidean_Distance_To_Hydrology",
        "Absolute_Hydrology_Elevation",
        "Mean_Distance_To_Amenities",
    ]

    # Combine and verify existence in the dataframe
    all_cont_cols = [
        c for c in base_cont_cols + new_cont_cols if c in X_train_df.columns
    ]

    # All other columns are treated as binary (Soil Types, Wilderness Areas)
    bin_cols = [c for c in X_train_df.columns if c not in all_cont_cols]

    # Standardize Continuous Features
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_df[all_cont_cols])
    X_val_cont = scaler.transform(X_val_df[all_cont_cols])
    X_test_cont = scaler.transform(X_test_df[all_cont_cols])

    # Process Binary Features (ensure float32 for PyTorch)
    X_train_bin = X_train_df[bin_cols].values.astype(np.float32)
    X_val_bin = X_val_df[bin_cols].values.astype(np.float32)
    X_test_bin = X_test_df[bin_cols].values.astype(np.float32)

    # Concatenate Continuous and Binary features
    X_train = np.hstack([X_train_cont, X_train_bin]).astype(np.float32)
    X_val = np.hstack([X_val_cont, X_val_bin]).astype(np.float32)
    X_test = np.hstack([X_test_cont, X_test_bin]).astype(np.float32)

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def get_datasets(load_cached_data=True, debug=False):
    """
    Loads data, applies feature engineering and preprocessing, handles caching,
    and returns PyTorch TensorDatasets.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
        debug (bool): If True, loads a subset of data for debugging.

    Returns:
        train_dataset (TensorDataset): Training data.
        val_dataset (TensorDataset): Validation data.
        test_dataset (TensorDataset): Test data.
        test_ids (np.array): IDs for the test set.
        classes (np.array): Class labels corresponding to target integers.
    """
    # Ensure working directories exist
    Config.create_directories()

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
        "meta": os.path.join(Config.CACHE_DIR, "meta.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["train_X"])
        y_train = np.load(cache_files["train_y"])
        X_val = np.load(cache_files["val_X"])
        y_val = np.load(cache_files["val_y"])
        X_test = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
        # Load metadata (classes)
        meta = np.load(cache_files["meta"], allow_pickle=True).item()
        classes = meta["classes"]
    else:
        print("Processing data from scratch...")

        # Load Raw Data from Parquet Metadata
        df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
        df_val = pd.read_parquet(Config.VAL_DATA_PATH)
        df_test = pd.read_parquet(Config.TEST_DATA_PATH)

        # Handle Debug Mode
        if debug:
            print(f"Debug mode enabled: using first {Config.DEBUG_SAMPLES} samples.")
            df_train = df_train.iloc[: Config.DEBUG_SAMPLES]
            df_val = df_val.iloc[: Config.DEBUG_SAMPLES]
            df_test = df_test.iloc[: Config.DEBUG_SAMPLES]

        # Apply Feature Engineering
        df_train = feature_engineering(df_train)
        df_val = feature_engineering(df_val)
        df_test = feature_engineering(df_test)

        # Apply Preprocessing
        X_train, y_train, X_val, y_val, X_test, test_ids, classes = preprocess_data(
            df_train, df_val, df_test
        )

        # Save to Cache
        np.save(cache_files["train_X"], X_train.astype(np.float32))
        np.save(cache_files["train_y"], y_train.astype(np.int64))
        np.save(cache_files["val_X"], X_val.astype(np.float32))
        np.save(cache_files["val_y"], y_val.astype(np.int64))
        np.save(cache_files["test_X"], X_test.astype(np.float32))
        np.save(cache_files["test_ids"], test_ids)
        np.save(cache_files["meta"], {"classes": classes})

    # Create PyTorch TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test))

    return train_dataset, val_dataset, test_dataset, test_ids, classes
