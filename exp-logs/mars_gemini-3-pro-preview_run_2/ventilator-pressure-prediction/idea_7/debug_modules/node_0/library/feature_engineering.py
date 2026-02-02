import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

from library.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COL,
    BREATH_ID_COL,
    TARGET_COL,
    SEED,
    DEBUG,
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds physics-based and temporal features to the dataframe.

    Features added:
    - u_in_cumsum: Cumulative sum of u_in (proxy for air volume).
    - R_u_in: Interaction term R * u_in (Resistive Pressure).
    - u_in_cumsum_div_C: Interaction term u_in_cumsum / C (Elastic Pressure).
    - u_in_lag[1-4]: Lagged values of u_in.
    - u_in_diff[1-4]: Difference between current u_in and lagged u_in.

    Args:
        df (pd.DataFrame): Raw dataframe containing 'breath_id', 'u_in', 'R', 'C'.

    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    # Ensure data is sorted by breath_id and time_step for correct lag/cumsum calculation
    # Assuming 'id' correlates with time_step within a breath, or use 'time_step'
    if "time_step" in df.columns:
        df = df.sort_values([BREATH_ID_COL, "time_step"])
    else:
        df = df.sort_values([BREATH_ID_COL, ID_COL])

    # --- Integral Features (Volume Proxy) ---
    # Explicitly generate u_in_cumsum to represent Air Volume
    # Groupby is necessary to reset cumsum for each breath
    df["u_in_cumsum"] = df.groupby(BREATH_ID_COL)["u_in"].cumsum()

    # --- Interaction Features (Physics) ---
    # Resistive Pressure ~ Flow * Resistance
    df["R_u_in"] = df["R"] * df["u_in"]

    # Elastic Pressure ~ Volume / Compliance
    df["u_in_cumsum_div_C"] = df["u_in_cumsum"] / df["C"]

    # --- Temporal Features (Derivatives/History) ---
    # Lags and Finite Differences
    # We use groupby shift to ensure lags do not cross breath boundaries
    for k in range(1, 5):
        lag_col = f"u_in_lag{k}"
        diff_col = f"u_in_diff{k}"

        # Lag feature
        df[lag_col] = df.groupby(BREATH_ID_COL)["u_in"].shift(k).fillna(0)

        # Difference feature (Current - Lag)
        # This captures the rate of change over k steps
        df[diff_col] = df["u_in"] - df[lag_col]

    # Fill any remaining NaNs (though fillna(0) above handles lags)
    df = df.fillna(0)

    return df


def get_processed_data(
    split: str, debug: bool = False, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads, processes, and caches data for a specific split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.
        debug (bool): If True, processes a small subset of the data.
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        pd.DataFrame: The processed dataframe with engineering features.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Construct cache filename
    debug_suffix = "_debug" if debug else ""
    cache_filename = f"processed_{split}{debug_suffix}.parquet"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {split} data (Debug={debug})...")

    # 2. Determine Metadata and Source File
    if split == "train":
        meta_path = TRAIN_META_PATH
        source_data_path = TRAIN_DATA_PATH
    elif split == "val":
        meta_path = VAL_META_PATH
        source_data_path = TRAIN_DATA_PATH  # Val is a subset of train.csv
    elif split == "test":
        meta_path = TEST_META_PATH
        source_data_path = TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # 3. Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Handle Debug Mode: Subsample metadata
    if debug:
        unique_breaths = df_meta[BREATH_ID_COL].unique()
        # Select first 100 breaths for deterministic debug set
        debug_breaths = unique_breaths[:100]
        df_meta = df_meta[df_meta[BREATH_ID_COL].isin(debug_breaths)]
        print(f"Debug mode: Filtered to {len(debug_breaths)} breaths.")

    target_breath_ids = set(df_meta[BREATH_ID_COL].unique())

    # 4. Load Raw Data
    # We load the full raw file and filter.
    # For very large files, chunking would be better, but dataset fits in memory.
    print(f"Loading raw data from {source_data_path}...")
    df_raw = pd.read_csv(source_data_path)

    # 5. Filter Data based on Metadata
    # This separates Train from Val, or selects Test
    df_filtered = df_raw[df_raw[BREATH_ID_COL].isin(target_breath_ids)].copy()

    # 6. Feature Engineering
    print("Applying feature engineering...")
    df_processed = add_features(df_filtered)

    # 7. Cache the result
    print(f"Saving processed data to {cache_path}...")
    df_processed.to_parquet(cache_path, index=False)

    return df_processed


def fit_scaler(df: pd.DataFrame) -> RobustScaler:
    """
    Fits a RobustScaler on the continuous features of the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing training data.

    Returns:
        RobustScaler: Fitted scaler object.
    """
    print("Fitting RobustScaler...")
    scaler = RobustScaler()
    scaler.fit(df[CONTINUOUS_FEATURES])
    return scaler


def save_scaler(scaler: RobustScaler, debug: bool = False):
    """
    Saves the fitted scaler to the cache directory.
    """
    debug_suffix = "_debug" if debug else ""
    path = os.path.join(CACHE_DIR, f"scaler{debug_suffix}.joblib")
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(debug: bool = False) -> RobustScaler:
    """
    Loads the scaler from the cache directory.
    """
    debug_suffix = "_debug" if debug else ""
    path = os.path.join(CACHE_DIR, f"scaler{debug_suffix}.joblib")
    if os.path.exists(path):
        print(f"Loading scaler from {path}")
        return joblib.load(path)
    return None


def transform_data(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """
    Applies the scaler to the continuous features of the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to transform.
        scaler (RobustScaler): Fitted scaler.

    Returns:
        pd.DataFrame: Dataframe with scaled features.
    """
    df_scaled = df.copy()
    df_scaled[CONTINUOUS_FEATURES] = scaler.transform(df[CONTINUOUS_FEATURES])
    return df_scaled
