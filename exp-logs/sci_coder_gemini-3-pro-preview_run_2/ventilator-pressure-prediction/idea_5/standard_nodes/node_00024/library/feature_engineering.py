import os
import pandas as pd
import numpy as np
from library.config import Config


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds physics-informed features based on the Equation of Motion.

    Includes:
    - Cumulative Volume (Integral of flow)
    - Interaction terms: R * Flow, Volume / C
    """
    # Ensure data is sorted for correct cumsum/shift operations
    df = df.sort_values(by=[Config.BREATH_ID_COL, "time_step"]).reset_index(drop=True)

    # 1. Cumulative Volume (u_in_cumsum)
    # We use groupby to ensure cumsum resets for each breath
    if Config.USE_CUMSUM:
        df["u_in_cumsum"] = df.groupby(Config.BREATH_ID_COL)["u_in"].cumsum()

    # 2. Interaction Terms
    if Config.USE_INTERACTIONS:
        # Resistive Pressure Component proxy: R * u_in
        df["R_mult_u_in"] = df["R"] * df["u_in"]

        # Elastic Pressure Component proxy: Volume / C
        # Note: We use the calculated cumsum as the volume proxy
        if "u_in_cumsum" in df.columns:
            df["u_in_cumsum_div_C"] = df["u_in_cumsum"] / df["C"]
        else:
            # Fallback if cumsum wasn't enabled, though it should be
            temp_cumsum = df.groupby(Config.BREATH_ID_COL)["u_in"].cumsum()
            df["u_in_cumsum_div_C"] = temp_cumsum / df["C"]

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal features to capture dynamics for the Instantaneous Path.

    Includes:
    - Lag features (History)
    - Finite Differences (Velocity/Acceleration proxies)
    """
    # Ensure data is sorted
    df = df.sort_values(by=[Config.BREATH_ID_COL, "time_step"]).reset_index(drop=True)

    # Group object for efficient shifting
    grouped_u_in = df.groupby(Config.BREATH_ID_COL)["u_in"]

    # 1. Lag Features
    if Config.USE_LAGS:
        for lag in Config.LAG_STEPS:
            df[f"u_in_lag{lag}"] = grouped_u_in.shift(lag).fillna(0)

    # 2. Finite Differences (Derivatives)
    # 1st Derivative (Velocity of flow change)
    if Config.USE_DIFFS:
        # We calculate diffs within groups to avoid boundary bleeding
        # diff(1) is u_in[t] - u_in[t-1]
        df["u_in_diff1"] = grouped_u_in.diff(1).fillna(0)

        # 2nd Derivative (Acceleration of flow change)
        if 2 in Config.DIFF_STEPS:
            df["u_in_diff2"] = grouped_u_in.diff(2).fillna(0)
            # Alternatively, diff of diff for true 2nd derivative approximation
            # df["u_in_accel"] = df.groupby(Config.BREATH_ID_COL)["u_in_diff1"].diff(1).fillna(0)

    # 3. Time delta (optional, but useful for physics consistency)
    # df["dt"] = df.groupby(Config.BREATH_ID_COL)["time_step"].diff(1).fillna(0)

    return df


def load_and_process_data(
    split_name: str, load_cached_data: bool = True, debug: bool = False
) -> pd.DataFrame:
    """
    Main entry point for data loading and feature engineering.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache first.
        debug (bool): If True, processes a smaller subset of data.

    Returns:
        pd.DataFrame: The processed dataframe with engineering features.
    """
    # 1. Setup Cache Path
    Config.setup()  # Ensure working directory exists

    cache_filename = f"processed_{split_name}"
    if debug:
        cache_filename += "_debug"
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_filename}.parquet")

    # 2. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {split_name} data from scratch...")

    # 3. Identify Metadata and Source Files
    if split_name == "train":
        meta_path = Config.TRAIN_METADATA
        source_csv = Config.TRAIN_CSV
    elif split_name == "val":
        meta_path = Config.VAL_METADATA
        source_csv = Config.TRAIN_CSV  # Val comes from train.csv
    elif split_name == "test":
        meta_path = Config.TEST_METADATA
        source_csv = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    # 4. Load Metadata
    df_meta = pd.read_csv(meta_path)

    if debug:
        # Sample breaths for debugging
        unique_breaths = df_meta[Config.BREATH_ID_COL].unique()
        sample_breaths = unique_breaths[:100]  # Take first 100 breaths
        df_meta = df_meta[df_meta[Config.BREATH_ID_COL].isin(sample_breaths)]

    target_breath_ids = df_meta[Config.BREATH_ID_COL].unique()

    # 5. Load Raw Data
    # We load the full source file (or necessary columns) and filter
    # Note: Reading full CSV is memory intensive but safe.
    # Optimization: Filter by chunks or use usecols if needed, but 5GB fits in 220GB RAM.
    print(f"Loading raw data from {source_csv}...")
    df_raw = pd.read_csv(source_csv)

    # 6. Filter and Merge
    # We filter the raw data to only include breaths present in the metadata for this split
    df_processed = df_raw[df_raw[Config.BREATH_ID_COL].isin(target_breath_ids)].copy()

    # Ensure sorting
    df_processed = df_processed.sort_values(
        by=[Config.BREATH_ID_COL, "time_step"]
    ).reset_index(drop=True)

    # 7. Feature Engineering
    print("Generating physics features...")
    df_processed = add_physics_features(df_processed)

    print("Generating temporal features...")
    df_processed = add_temporal_features(df_processed)

    # 8. Type Optimization (Optional but good for cache size)
    # Convert float64 to float32 to save space/memory
    float_cols = df_processed.select_dtypes(include=["float64"]).columns
    df_processed[float_cols] = df_processed[float_cols].astype("float32")

    # 9. Cache Data
    print(f"Saving processed data to {cache_path}...")
    df_processed.to_parquet(cache_path, index=False)

    return df_processed
