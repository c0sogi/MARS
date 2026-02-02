import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SCALER_CENTER_PATH,
    SCALER_SCALE_PATH,
    FEATURE_NAMES,
)


def engineer_features(df):
    """
    Performs physics-fidelity feature engineering on the provided DataFrame.

    Args:
        df (pd.DataFrame): Raw dataframe containing breath data.

    Returns:
        pd.DataFrame: Dataframe with added engineering features.
    """
    # 1. Calculate Time Delta (dt)
    # Group by breath_id to ensure we don't diff across breaths.
    # Fill NA with 0 for the first time step of each breath.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Volume Integration
    # Volume is the cumulative sum of flow (u_in) * dt
    df["dV"] = df["u_in"] * df["dt"]
    df["volume"] = df.groupby("breath_id")["dV"].cumsum()

    # 3. Temporal Dynamics (Lags)
    # Generate lags 1-4 for u_in
    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # 4. Temporal Dynamics (Differences)
    # First and Second derivatives of control input
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff(1).fillna(0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in"].diff(2).fillna(0)

    # 5. Soft Physics Interactions
    # u_in * R (Resistive pressure component proxy)
    df["u_in_R"] = df["u_in"] * df["R"]
    # volume / C (Elastic pressure component proxy)
    df["vol_C"] = df["volume"] / df["C"]

    # Cleanup intermediate columns
    df.drop(columns=["dt", "dV"], inplace=True, errors="ignore")

    return df


def prepare_datasets(load_cached_data=True):
    """
    Orchestrates the loading, engineering, scaling, and caching of datasets.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df) processed and scaled.
    """
    # Check if all required cache files exist
    caches_exist = (
        os.path.exists(TRAIN_CACHE_PATH)
        and os.path.exists(VAL_CACHE_PATH)
        and os.path.exists(TEST_CACHE_PATH)
        and os.path.exists(SCALER_CENTER_PATH)
        and os.path.exists(SCALER_SCALE_PATH)
    )

    if load_cached_data and caches_exist:
        print("Loading cached datasets and scaler artifacts...")
        train_df = pd.read_parquet(TRAIN_CACHE_PATH)
        val_df = pd.read_parquet(VAL_CACHE_PATH)
        test_df = pd.read_parquet(TEST_CACHE_PATH)
        return train_df, val_df, test_df

    print("Cache not found or reload requested. Processing from scratch...")

    # 1. Load Raw Data
    print(f"Loading raw data from {os.path.dirname(TRAIN_DATA_PATH)}...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # 2. Engineer Features
    print("Engineering features for Training set...")
    train_df = engineer_features(train_df)

    print("Engineering features for Validation set...")
    val_df = engineer_features(val_df)

    print("Engineering features for Test set...")
    test_df = engineer_features(test_df)

    # 3. Scaling
    print("Fitting RobustScaler on Training features...")
    scaler = RobustScaler(
        quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True
    )

    # Fit strictly on Train data to avoid leakage
    scaler.fit(train_df[FEATURE_NAMES])

    # Save scaler statistics manually as requested
    np.save(SCALER_CENTER_PATH, scaler.center_)
    np.save(SCALER_SCALE_PATH, scaler.scale_)

    # Transform all datasets
    print("Applying scaling to all datasets...")
    train_df[FEATURE_NAMES] = scaler.transform(train_df[FEATURE_NAMES])
    val_df[FEATURE_NAMES] = scaler.transform(val_df[FEATURE_NAMES])
    test_df[FEATURE_NAMES] = scaler.transform(test_df[FEATURE_NAMES])

    # 4. Save to Cache
    print(f"Saving processed datasets to {os.path.dirname(TRAIN_CACHE_PATH)}...")
    os.makedirs(os.path.dirname(TRAIN_CACHE_PATH), exist_ok=True)

    train_df.to_parquet(TRAIN_CACHE_PATH, index=False)
    val_df.to_parquet(VAL_CACHE_PATH, index=False)
    test_df.to_parquet(TEST_CACHE_PATH, index=False)

    print("Data preparation complete.")
    return train_df, val_df, test_df
