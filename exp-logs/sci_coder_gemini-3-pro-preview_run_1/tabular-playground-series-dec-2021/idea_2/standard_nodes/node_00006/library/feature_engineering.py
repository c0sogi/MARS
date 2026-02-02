import os
import numpy as np
import pandas as pd
from library.config import Config


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds physics-informed interaction features to the dataframe.

    Features added:
    1. Euclidean_Distance_To_Hydrology: sqrt(Horizontal_Dist^2 + Vertical_Dist^2)
    2. Hydrology_Elevation: Elevation - Vertical_Distance_To_Hydrology

    Args:
        df: Input DataFrame containing raw features.

    Returns:
        DataFrame with added features.
    """
    # Create a copy to avoid SettingWithCopy warnings on slices
    df = df.copy()

    # Ensure required columns exist
    required_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Elevation",
    ]

    if not all(col in df.columns for col in required_cols):
        # If columns are missing (unlikely given the dataset), return as is or raise error.
        # Here we proceed assuming data integrity based on metadata analysis.
        return df

    # Feature 1: Euclidean Distance to Hydrology
    # Represents the straight-line distance to water sources, combining horizontal and vertical components.
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Feature 2: Hydrology Elevation
    # Represents the absolute elevation of the hydrology source.
    # If Elevation is point elevation and Vertical_Dist is (Elevation - Hydro_Elevation),
    # then Hydro_Elevation = Elevation - Vertical_Dist.
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    return df


def process_and_cache_data(load_cached_data: bool = True):
    """
    Loads data, applies feature engineering, and manages caching using Parquet files.

    Logic:
    1. Checks if cached files exist in Config.IDEA_DIR.
    2. If load_cached_data=True and files exist, loads and returns them.
    3. Otherwise, loads raw data from Config metadata paths, applies feature engineering,
       saves to cache, and returns them.

    Args:
        load_cached_data: Boolean flag to enable/disable loading from cache.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure output directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Define cache file paths
    train_cache_path = os.path.join(Config.IDEA_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(Config.IDEA_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(Config.IDEA_DIR, "test_processed.parquet")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        try:
            train_df = pd.read_parquet(train_cache_path)
            val_df = pd.read_parquet(val_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return train_df, val_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")
            # Fall through to processing logic

    print("Processing data from scratch...")

    # Load raw data using paths from Config
    # Using pandas read_csv as per metadata instructions
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Apply Feature Engineering
    print("Applying interaction features...")
    train_df = add_interaction_features(train_df)
    val_df = add_interaction_features(val_df)
    test_df = add_interaction_features(test_df)

    # Save to cache
    print(f"Saving processed data to {Config.IDEA_DIR}...")
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, val_df, test_df
