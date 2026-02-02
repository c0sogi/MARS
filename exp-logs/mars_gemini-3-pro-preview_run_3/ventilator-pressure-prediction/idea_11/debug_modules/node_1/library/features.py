import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config


def add_features(df):
    """
    Adds PID, Lookahead, and Physics interaction features to the dataframe.
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # Group by breath_id for window operations
    # Note: Using groupby().shift() can be slow on large dfs.
    # Since we sorted, we can use vectorized operations with masks,
    # but for safety and readability within the 24h constraint, we use groupby
    # which is robust and sufficiently fast for 6M rows (~1-2 mins).
    grp = df.groupby("breath_id")

    # --- PID Features ---
    # Integral (Volume proxy)
    df["u_in_cumsum"] = grp["u_in"].cumsum()

    # Derivative (Flow change)
    df["u_in_diff1"] = grp["u_in"].diff().fillna(0)

    # Acceleration
    df["u_in_diff2"] = grp["u_in_diff1"].diff().fillna(0)

    # --- Non-Causal Lookahead Features ---
    # Future control inputs (allowed in offline task)
    df["u_in_next"] = grp["u_in"].shift(-1).fillna(0)
    df["u_in_next2"] = grp["u_in"].shift(-2).fillna(0)

    # --- Physics Features ---
    # Time delta
    df["dt"] = grp["time_step"].diff().fillna(0)

    # Area (Volume approximation: Integral of u_in * dt)
    # u_in is 0-100, dt is time.
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # Interaction: Resistance * Flow (u_in) -> Resistive Pressure component
    df["R_u_in"] = df["R"] * df["u_in"]

    # Interaction: Volume / Compliance -> Elastic Pressure component
    # Avoid division by zero if C is ever 0 (it's 10, 20, 50 in this dataset, but good practice)
    df["area_C"] = df["area"] / df["C"]

    # Cleanup temporary columns if any (dt is useful, keep it if needed, but not in Config list)
    # Config.DYN_FEATURES doesn't list 'dt', so it will be dropped during selection.

    return df


def reshape_to_sequences(df, feature_cols, target_col=None):
    """
    Reshapes flat dataframe to (N_breaths, 80, N_features).
    Assumes df is sorted by breath_id and time_step.
    """
    # 80 time steps per breath
    SEQ_LEN = 80

    # Extract features
    data_x = df[feature_cols].values
    num_rows = data_x.shape[0]

    if num_rows % SEQ_LEN != 0:
        raise ValueError(
            f"Total rows {num_rows} is not divisible by sequence length {SEQ_LEN}"
        )

    num_breaths = num_rows // SEQ_LEN

    # Reshape X: (N, 80, F)
    data_x = data_x.reshape(num_breaths, SEQ_LEN, len(feature_cols))

    data_y = None
    if target_col and target_col in df.columns:
        data_y = df[target_col].values
        # Reshape Y: (N, 80)
        data_y = data_y.reshape(num_breaths, SEQ_LEN)

    return data_x, data_y


def prepare_datasets(load_cached_data=True):
    """
    Main function to load, process, scale, and reshape data.
    Implements caching to save time on re-runs.
    """
    # 1. Check Cache
    cache_files = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        Config.CACHE_TEST_X,
        Config.CACHE_TEST_IDS,
        Config.SCALER_PATH,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        train_x = np.load(Config.CACHE_TRAIN_X)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        val_x = np.load(Config.CACHE_VAL_X)
        val_y = np.load(Config.CACHE_VAL_Y)
        test_x = np.load(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS)
        return train_x, train_y, val_x, val_y, test_x, test_ids

    print("Cache not found or reload requested. Processing datasets...")

    # 2. Load Metadata CSVs
    print(f"Loading train from {Config.TRAIN_CSV}")
    train_df = pd.read_csv(Config.TRAIN_CSV)

    print(f"Loading val from {Config.VAL_CSV}")
    val_df = pd.read_csv(Config.VAL_CSV)

    print(f"Loading test from {Config.TEST_CSV}")
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Feature Engineering
    print("Generating features...")
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    # 4. Scaling
    # We scale Dynamic and Static features. We do NOT scale Control features (u_out is binary).
    cols_to_scale = Config.DYN_FEATURES + Config.STATIC_FEATURES

    print(f"Fitting RobustScaler on {len(cols_to_scale)} features...")
    scaler = RobustScaler()

    # Fit on Train only
    scaler.fit(train_df[cols_to_scale])

    # Transform all
    train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
    val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    # Save scaler stats for reproducibility/inference
    np.savez(Config.SCALER_PATH, center=scaler.center_, scale=scaler.scale_)

    # 5. Reshaping and Selection
    print("Reshaping to 3D tensors...")
    feature_cols = Config.FEATURE_COLS  # Order defined in Config

    train_x, train_y = reshape_to_sequences(train_df, feature_cols, Config.TARGET_COL)
    val_x, val_y = reshape_to_sequences(val_df, feature_cols, Config.TARGET_COL)
    test_x, _ = reshape_to_sequences(test_df, feature_cols, None)

    # Extract Test IDs (needed for submission)
    # We need one ID per row in the final submission, but the processing is done per breath.
    # However, the submission format is flat.
    # We will store the full flat IDs or reshaped IDs?
    # The submission generation usually iterates over the test set.
    # Let's save the flat IDs array corresponding to the test_x sequence.
    # Actually, reshape_to_sequences returns (N, 80, F).
    # Let's save the IDs in (N, 80) shape to match.
    test_ids = test_df[Config.ID_COL].values.reshape(-1, 80)

    # 6. Save to Cache
    print("Saving to cache...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    np.save(Config.CACHE_TRAIN_X, train_x)
    np.save(Config.CACHE_TRAIN_Y, train_y)
    np.save(Config.CACHE_VAL_X, val_x)
    np.save(Config.CACHE_VAL_Y, val_y)
    np.save(Config.CACHE_TEST_X, test_x)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    print(f"Dataset processing complete.")
    print(f"Train X shape: {train_x.shape}")
    print(f"Val X shape:   {val_x.shape}")
    print(f"Test X shape:  {test_x.shape}")

    return train_x, train_y, val_x, val_y, test_x, test_ids
