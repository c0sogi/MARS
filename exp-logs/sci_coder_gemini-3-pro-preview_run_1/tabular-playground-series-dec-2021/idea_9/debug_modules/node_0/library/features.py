import os
import numpy as np
import pandas as pd
from library import config


def add_geometry_features(df):
    """
    Computes physics-informed geometric features.

    Adds:
    - Euclidean_Distance_To_Hydrology: sqrt(H_Dist^2 + V_Dist^2)
    - Elevation_Hydrology: Elevation - Vertical_Distance_To_Hydrology
    - Aspect_Sin: sin(Aspect)
    - Aspect_Cos: cos(Aspect)
    """
    # Euclidean Distance to Hydrology
    # H = Horizontal_Distance_To_Hydrology, V = Vertical_Distance_To_Hydrology
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Relative Elevation (Elevation of the hydrology source)
    # Vertical_Distance_To_Hydrology = Elevation - Elevation_Hydrology
    # Therefore: Elevation_Hydrology = Elevation - Vertical_Distance_To_Hydrology
    df["Elevation_Hydrology"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # Cyclic Aspect encoding (converting degrees to radians first)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    return df


def add_dense_categorical_features(df):
    """
    Generates dense integer indices for One-Hot Encoded columns while retaining originals.

    - Sorts OHE columns numerically (e.g., Soil_Type1, Soil_Type2...) to preserve semantics.
    - Creates 'Soil_Type_Index' and 'Wilderness_Area_Index'.
    """
    # Process Soil_Type
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    # Explicitly sort numerically by extracting the integer suffix
    soil_cols.sort(key=lambda x: int(x.replace("Soil_Type", "")))

    if soil_cols:
        # Use argmax to find the index of the active column (0-39)
        # This assumes rows are strictly One-Hot (or Zero-Hot).
        # For this dataset, it provides a strong ordinal signal for tree splits.
        df["Soil_Type_Index"] = np.argmax(df[soil_cols].values, axis=1)

    # Process Wilderness_Area
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    wild_cols.sort(key=lambda x: int(x.replace("Wilderness_Area", "")))

    if wild_cols:
        df["Wilderness_Area_Index"] = np.argmax(df[wild_cols].values, axis=1)

    return df


def get_processed_data(split_name, load_cached_data=True):
    """
    Loads, processes, and caches data for a specific split.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(config.CACHE_DIR, f"{split_name}_processed.parquet")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split_name} data from scratch...")

    # Load raw data based on split
    if split_name == "train":
        df = pd.read_csv(config.TRAIN_PATH)
    elif split_name == "val":
        df = pd.read_csv(config.VAL_PATH)
    elif split_name == "test":
        df = pd.read_csv(config.TEST_PATH)
    else:
        raise ValueError(f"Unknown split name: {split_name}")

    # Apply Feature Engineering
    df = add_geometry_features(df)
    df = add_dense_categorical_features(df)

    # Map Target if present (Train/Val)
    if config.TARGET_COL in df.columns:
        print(f"Mapping target column '{config.TARGET_COL}'...")
        df[config.TARGET_COL] = df[config.TARGET_COL].map(config.TARGET_MAPPING)

        # Verify mapping didn't introduce NaNs
        if df[config.TARGET_COL].isnull().any():
            raise ValueError(
                "Target mapping resulted in NaN values. Check TARGET_MAPPING configuration."
            )

        # Ensure target is integer type
        df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)

    # Save to cache
    print(f"Saving processed {split_name} data to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df
