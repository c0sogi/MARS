import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from library.config import Config


def engineer_features(dataset_type: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Performs physics-fidelity feature engineering on the ventilator dataset.
    Handles caching, physics term calculation, temporal feature generation, and scaling.

    Args:
        dataset_type (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache first.

    Returns:
        pd.DataFrame: The processed dataframe ready for the model.
    """
    # 1. Determine paths based on dataset type
    if dataset_type == "train":
        input_path = Config.TRAIN_PATH
        cache_path = Config.CACHE_TRAIN_PATH
    elif dataset_type == "val":
        input_path = Config.VAL_PATH
        cache_path = Config.CACHE_VAL_PATH
    elif dataset_type == "test":
        input_path = Config.TEST_PATH
        cache_path = Config.CACHE_TEST_PATH
    else:
        raise ValueError(
            f"Invalid dataset_type: {dataset_type}. Must be 'train', 'val', or 'test'."
        )

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {dataset_type} data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {dataset_type} data from {input_path}...")

    # 3. Load Raw Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # 4. Physics Feature Engineering
    # Sort to ensure time order (though data should be sorted)
    df = df.sort_values([Config.BREATH_ID_COL, Config.TIME_COL]).reset_index(drop=True)

    # Calculate dt (time delta)
    # Group by breath_id to ensure the first step of a new breath doesn't diff with previous breath
    df["dt"] = df.groupby(Config.BREATH_ID_COL)[Config.TIME_COL].diff().fillna(0)

    # Calculate Volume: Integral of u_in * dt
    # u_in is 0-100, we treat it as flow rate.
    df["volume"] = (
        (df[Config.U_IN_COL] * df["dt"]).groupby(df[Config.BREATH_ID_COL]).cumsum()
    )

    # Interaction Terms (Equation of Motion proxies)
    # u_in * R
    df["u_in_R"] = df[Config.U_IN_COL] * df[Config.R_COL]
    # volume / C
    df["vol_C"] = df["volume"] / df[Config.C_COL]

    # 5. Temporal Features (Lags and Diffs)
    # Lags
    for lag in Config.LAG_STEPS:
        df[f"u_in_lag{lag}"] = (
            df.groupby(Config.BREATH_ID_COL)[Config.U_IN_COL].shift(lag).fillna(0)
        )

    # Diffs
    if Config.USE_DIFFS:
        df["u_in_diff1"] = (
            df.groupby(Config.BREATH_ID_COL)[Config.U_IN_COL].diff(1).fillna(0)
        )
        df["u_in_diff2"] = (
            df.groupby(Config.BREATH_ID_COL)[Config.U_IN_COL].diff(2).fillna(0)
        )

    # 6. Scaling
    # We only scale continuous features. u_out must remain binary.
    # Ensure scaler directory exists
    os.makedirs(os.path.dirname(Config.SCALER_PATH), exist_ok=True)

    if dataset_type == "train":
        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler()
        df[Config.CONT_FEATURES] = scaler.fit_transform(df[Config.CONT_FEATURES])
        # Save scaler
        joblib.dump(scaler, Config.SCALER_PATH)
    else:
        # Load scaler
        if not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler not found at {Config.SCALER_PATH}. Process 'train' first."
            )

        print("Loading fitted RobustScaler...")
        scaler = joblib.load(Config.SCALER_PATH)
        df[Config.CONT_FEATURES] = scaler.transform(df[Config.CONT_FEATURES])

    # 7. Final Column Selection and Type Optimization
    # We keep identifiers, target (if exists), continuous features, and binary features
    cols_to_keep = [Config.ID_COL, Config.BREATH_ID_COL]

    if Config.TARGET_COL in df.columns:
        cols_to_keep.append(Config.TARGET_COL)

    cols_to_keep.extend(Config.CONT_FEATURES)
    cols_to_keep.extend(Config.BINARY_FEATURES)

    df = df[cols_to_keep]

    # Downcast floats to float32 to save memory
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # 8. Save to Cache
    print(f"Saving processed {dataset_type} data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df
