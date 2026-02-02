import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("features")


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates physics-informed features:
    - Euclidean Distance to Hydrology: sqrt(H^2 + V^2)
    - Hydrology Elevation: Elevation - Vertical_Distance_To_Hydrology
    - Cyclic Aspect: Sin and Cos of Aspect
    """
    # Ensure required columns exist
    required_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Elevation",
        "Aspect",
    ]
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing columns for physics features. Available: {df.columns}")
        return df

    # Euclidean Distance to Hydrology
    h_dist = df["Horizontal_Distance_To_Hydrology"]
    v_dist = df["Vertical_Distance_To_Hydrology"]
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_dist**2 + v_dist**2)

    # Hydrology Elevation (Absolute elevation of the water source)
    # Vertical_Dist = Elevation - Hydro_Elevation => Hydro_Elevation = Elevation - Vertical_Dist
    df["Hydrology_Elevation"] = df["Elevation"] - v_dist

    # Cyclic Aspect
    # Convert Aspect (degrees) to radians
    aspect_rad = np.deg2rad(df["Aspect"])
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    return df


def engineer_dense_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates dense integer indices for Soil_Type and Wilderness_Area
    while retaining the original OHE columns.

    Logic: Dot product of OHE columns with [1, 2, ..., N].
    """
    # 1. Soil_Type (1 to 40)
    soil_cols = [f"Soil_Type{i}" for i in range(1, 41)]
    # Filter for columns that actually exist in the dataframe
    present_soil_cols = [c for c in soil_cols if c in df.columns]

    if present_soil_cols:
        # Create a multiplier vector [1, 2, ..., N]
        multipliers = np.arange(1, len(present_soil_cols) + 1)
        # Compute dot product
        df["Soil_Type_Index"] = df[present_soil_cols].dot(multipliers).astype(int)

    # 2. Wilderness_Area (1 to 4)
    wild_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]
    present_wild_cols = [c for c in wild_cols if c in df.columns]

    if present_wild_cols:
        multipliers = np.arange(1, len(present_wild_cols) + 1)
        df["Wilderness_Area_Index"] = df[present_wild_cols].dot(multipliers).astype(int)

    return df


def preprocess_data(
    load_cached_data: bool = True,
    debug: bool = Config.DEBUG,
    debug_samples: int = Config.DEBUG_SAMPLES,
):
    """
    Orchestrates the feature engineering pipeline.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, subsamples the data for rapid prototyping.
        debug_samples (int): Number of samples to use in debug mode.

    Returns:
        train_df, val_df, test_df
    """
    cache_dir = Config.IDEA_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # --- 1. Attempt Cache Load ---
    # We only load cache if NOT in debug mode.
    # If debug is True, we want to load raw and subsample fresh.
    if load_cached_data and not debug:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            logger.info(f"Loading processed data from cache: {cache_dir}")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")
        else:
            logger.info("Cache not found. Processing from scratch...")
    else:
        if debug:
            logger.info(
                f"Debug mode enabled. Processing fresh subsample of {debug_samples} rows."
            )
        else:
            logger.info("Cache loading disabled. Processing from scratch...")

    # --- 2. Load Raw Data ---
    logger.info("Loading raw data from metadata...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # --- 3. Debug Subsampling ---
    if debug:
        logger.info(f"Subsampling data to {debug_samples} samples...")
        train_df = train_df.sample(
            n=min(len(train_df), debug_samples), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_samples), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), debug_samples), random_state=Config.SEED
        ).reset_index(drop=True)

    # --- 4. Feature Engineering ---
    logger.info("Generating Physics-Informed Features...")
    train_df = engineer_physics_features(train_df)
    val_df = engineer_physics_features(val_df)
    test_df = engineer_physics_features(test_df)

    logger.info("Generating Dense Indices...")
    train_df = engineer_dense_indices(train_df)
    val_df = engineer_dense_indices(val_df)
    test_df = engineer_dense_indices(test_df)

    # --- 5. Save to Cache ---
    # Only save if NOT in debug mode to avoid overwriting full cache with subsample
    if not debug:
        logger.info(f"Saving processed data to {cache_dir}...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)
    else:
        logger.info("Debug mode: Skipping cache save.")

    return train_df, val_df, test_df
