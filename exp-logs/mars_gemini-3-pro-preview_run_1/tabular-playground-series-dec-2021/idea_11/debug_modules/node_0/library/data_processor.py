import os
import numpy as np
import pandas as pd
from library.config import Config


def engineer_static_features(df):
    """
    Applies physics-informed transformations and robust densification to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with new features.
    """
    # 1. Physics-Informed Features
    # Euclidean Distance to Hydrology: sqrt(H^2 + V^2)
    # Ensure columns exist to avoid errors
    if (
        "Horizontal_Distance_To_Hydrology" in df.columns
        and "Vertical_Distance_To_Hydrology" in df.columns
    ):
        h_dist = df["Horizontal_Distance_To_Hydrology"]
        v_dist = df["Vertical_Distance_To_Hydrology"]
        df["Hydrology_Distance"] = np.sqrt(h_dist**2 + v_dist**2)

        # Relative Elevation: Elevation - Vertical_Dist
        if "Elevation" in df.columns:
            df["Relative_Elevation"] = df["Elevation"] - v_dist

    # Cyclic Aspect
    if "Aspect" in df.columns:
        # Convert degrees to radians
        aspect_rad = np.radians(df["Aspect"])
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Robust Densification (Dot Product)
    # Soil_Type
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    if soil_cols:
        # Create index vector [1, 2, ..., N]
        soil_indices = np.arange(1, len(soil_cols) + 1)
        # Dot product: If one-hot, result is the index. If all zero (missing), result is 0.
        # This handles the "missing" category implicitly by mapping it to 0.
        df["Soil_Type_Index"] = df[soil_cols].dot(soil_indices)

    # Wilderness_Area
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    if wild_cols:
        wild_indices = np.arange(1, len(wild_cols) + 1)
        df["Wilderness_Area_Index"] = df[wild_cols].dot(wild_indices)

    return df


def load_data():
    """
    Loads the raw training and test data from the paths specified in Config.

    Returns:
        tuple: (train_df, test_df)
    """
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"{Config.TRAIN_PATH} not found.")
    if not os.path.exists(Config.TEST_PATH):
        raise FileNotFoundError(f"{Config.TEST_PATH} not found.")

    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    return train_df, test_df


def process_data(load_cached_data=True):
    """
    Orchestrates the data loading and processing pipeline with caching.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (train_df, test_df)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        try:
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return train_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # Load raw data
    train_df, test_df = load_data()

    # Engineer features
    print("Engineering static features for training set...")
    train_df = engineer_static_features(train_df)

    print("Engineering static features for test set...")
    test_df = engineer_static_features(test_df)

    # Save to cache
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    train_df.to_parquet(train_cache_path)
    test_df.to_parquet(test_cache_path)

    return train_df, test_df
