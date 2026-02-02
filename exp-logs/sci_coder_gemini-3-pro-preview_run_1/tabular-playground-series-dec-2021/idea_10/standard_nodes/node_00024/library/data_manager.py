import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import setup_logger, save_to_cache, load_from_cache

# Initialize logger
logger = setup_logger("data_manager")


def add_geometric_features(df):
    """
    Adds physics-informed geometric features to the DataFrame.

    Features added:
    - Euclidean_Distance_To_Hydrology: sqrt(H_Dist^2 + V_Dist^2)
    - Elevation_Minus_VDH: Elevation - Vertical_Distance_To_Hydrology
    - Aspect_Sin: sin(Aspect_radians)
    - Aspect_Cos: cos(Aspect_radians)
    """
    logger.info("Adding geometric features...")

    # Euclidean Distance to Hydrology
    # Represents the straight-line distance to the water source
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Relative Elevation
    # Represents the absolute elevation of the hydrology source relative to the point
    df["Elevation_Minus_VDH"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # Cyclic Aspect
    # Convert Aspect (degrees) to radians for cyclic encoding
    aspect_rad = np.deg2rad(df["Aspect"])
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    return df


def create_dense_categoricals(df):
    """
    Creates dense integer indices from One-Hot Encoded (OHE) columns.
    Retains the original OHE columns.

    The function identifies columns based on prefixes defined in Config,
    sorts them numerically (e.g., Type1, Type2, ... Type10), and computes
    a dot product to generate a single integer index column.
    """
    logger.info("Creating dense categorical features...")

    prefixes = Config.CATEGORICAL_PREFIXES

    for prefix in prefixes:
        # Identify columns belonging to this prefix
        # Filter columns that start with prefix and follow with a number
        relevant_cols = [
            c for c in df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()
        ]

        if not relevant_cols:
            logger.warning(f"No columns found for prefix '{prefix}'")
            continue

        # Sort columns numerically based on the suffix integer
        # This ensures Soil_Type10 comes after Soil_Type9, not Soil_Type1
        relevant_cols.sort(key=lambda x: int(x[len(prefix) :]))

        # Create dense index (1-based)
        # We use dot product with [1, 2, ..., N]
        indices = np.arange(1, len(relevant_cols) + 1)

        dense_col_name = f"{prefix}_Index"

        # Matrix multiplication for efficient calculation
        # Result is 0 if all OHE columns are 0 (though unlikely in this dataset)
        df[dense_col_name] = df[relevant_cols].dot(indices).astype(np.int32)

        logger.info(
            f"Created '{dense_col_name}' from {len(relevant_cols)} OHE columns."
        )

    return df


def load_data(load_cached_data=True):
    """
    Loads training and test data.

    Implements a caching mechanism:
    1. Checks if processed parquet files exist in the cache directory.
    2. If found and load_cached_data is True, loads and returns them.
    3. If not found, loads raw CSVs from metadata, applies feature engineering,
       saves to cache, and returns the processed DataFrames.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, test_df)
    """
    train_cache_path = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        logger.info(f"Checking cache at: {Config.CACHE_DIR}")
        train_df = load_from_cache(train_cache_path)
        test_df = load_from_cache(test_cache_path)

        if train_df is not None and test_df is not None:
            logger.info("Successfully loaded processed data from cache.")
            return train_df, test_df
        else:
            logger.info("Cache miss (files not found). Proceeding to load raw data.")

    # 2. Load from source
    logger.info("Loading raw data from CSVs...")

    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Train file not found: {Config.TRAIN_CSV}")
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test file not found: {Config.TEST_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    logger.info(f"Raw Train shape: {train_df.shape}")
    logger.info(f"Raw Test shape: {test_df.shape}")

    # 3. Process Data
    # Apply geometric features
    if Config.ENABLE_GEOMETRIC_FEATURES:
        train_df = add_geometric_features(train_df)
        test_df = add_geometric_features(test_df)

    # Create dense categoricals (required for Target Encoding later in the pipeline)
    train_df = create_dense_categoricals(train_df)
    test_df = create_dense_categoricals(test_df)

    # 4. Save to cache
    logger.info("Saving processed data to cache...")
    save_to_cache(train_df, train_cache_path)
    save_to_cache(test_df, test_cache_path)

    return train_df, test_df
