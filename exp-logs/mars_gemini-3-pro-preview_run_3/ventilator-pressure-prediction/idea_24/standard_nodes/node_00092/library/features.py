import pandas as pd
import numpy as np
import os
from library.config import (
    CACHE_DIR,
    FEATURE_GENERATION_CONFIG,
    MODEL_FEATURES,
    EXCLUDED_FEATURES,
)


def engineer_features(df_path, output_name, load_cached_data=True):
    """
    Engineers features for the ventilator pressure prediction task.
    Implements the Kinematically-Augmented Residual-Hybrid (KARH-Net) feature set.

    Args:
        df_path (str): Path to the input CSV file.
        output_name (str): Name for the cached file (e.g., 'train_features').
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Dataframe with engineered features.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{output_name}.parquet")

    # 1. Cache Check
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {output_name}...")

    # 2. Load Data
    df = pd.read_csv(df_path)

    # 3. Feature Engineering
    # We use vectorized operations with boundary masking to ensure group-wise correctness
    # without the overhead of groupby().apply().

    # Identify breath boundaries
    # breath_id_lag: breath_id at t-1
    # breath_id_lead: breath_id at t+1
    df["breath_id_lag"] = df["breath_id"].shift(1)
    df["breath_id_lead"] = df["breath_id"].shift(-1)

    # Masks for valid transitions
    # start_mask: True if current row is the first step of a breath
    start_mask = df["breath_id"] != df["breath_id_lag"]
    # end_mask: True if current row is the last step of a breath
    end_mask = df["breath_id"] != df["breath_id_lead"]

    # --- Time Delta (dt) ---
    df["time_step_diff"] = df["time_step"].diff()
    df.loc[start_mask, "time_step_diff"] = 0.0

    # --- Kinematics: Backward (Momentum) ---
    # u_in_diff1: Velocity (1st derivative)
    df["u_in_diff1"] = df["u_in"].diff()
    df.loc[start_mask, "u_in_diff1"] = 0.0

    # u_in_diff2: Acceleration (2nd derivative)
    df["u_in_diff2"] = df["u_in_diff1"].diff()
    df.loc[start_mask, "u_in_diff2"] = 0.0

    # --- Kinematics: Forward (Intent) ---
    # u_in_fwd1: Forward Velocity (t+1 - t)
    df["u_in_fwd1"] = df["u_in"].shift(-1) - df["u_in"]
    df.loc[end_mask, "u_in_fwd1"] = 0.0

    # --- Lookahead Context ---
    # Leads 1 to 4 as defined in config
    for i in range(1, FEATURE_GENERATION_CONFIG["leads"] + 1):
        col_name = f"u_in_lead{i}"
        df[col_name] = df["u_in"].shift(-i)

        # Mask out values that shifted from the next breath
        # Check if breath_id at t is same as breath_id at t+i
        same_breath = df["breath_id"] == df["breath_id"].shift(-i)
        df.loc[~same_breath, col_name] = 0.0

    # --- Physical State ---
    # Area (Volume): Cumulative Integral of Flow
    # cumsum is optimized in pandas and respects groups efficiently
    df["area"] = (df["time_step_diff"] * df["u_in"]).groupby(df["breath_id"]).cumsum()

    # Interactions
    df["R_u_in"] = df["R"] * df["u_in"]
    df["area_C"] = df["area"] / df["C"]

    # 4. Cleanup and Selection
    feature_cols = MODEL_FEATURES

    # Validate features exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Failed to generate features: {missing_cols}")

    # Select columns: IDs + Target (if present) + Features
    keep_cols = ["id", "breath_id"]
    if "pressure" in df.columns:
        keep_cols.append("pressure")

    keep_cols.extend(feature_cols)

    # Create final dataframe
    df_out = df[keep_cols].copy()

    # Fill any remaining NaNs (e.g., from shifts) with 0
    df_out = df_out.fillna(0)

    # Cast features to float32 for memory efficiency
    # Exclude ID columns from casting
    float_cols = [c for c in df_out.columns if c not in ["id", "breath_id"]]
    df_out[float_cols] = df_out[float_cols].astype(np.float32)

    # 5. Save to Cache
    print(f"Saving features to {cache_path}...")
    df_out.to_parquet(cache_path, index=False)

    return df_out
