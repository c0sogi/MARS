import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


def add_features(df):
    """
    Adds physics-fidelity features and time-series dynamics to the dataframe.
    Implements the 'Physics-Fidelity' strategy:
    - Correct Volume Integration: sum(u_in * dt)
    - Soft Physics Interactions: u_in * R, volume / C
    - Dynamics: Lags and Derivatives
    """
    # Ensure data is sorted by breath and time to guarantee correct lag/diff calculations
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # --- 1. Time Delta and Volume Integration ---
    # Calculate dt (time difference).
    # We use a mask to handle the start of each breath where dt should be 0.
    df["dt"] = df["time_step"].diff().fillna(0)
    mask_start = df["breath_id"] != df["breath_id"].shift(1)
    df.loc[mask_start, "dt"] = 0

    # Volume = Integral of flow (u_in) over time
    # We calculate the area per step and then cumsum grouped by breath
    df["area"] = df["u_in"] * df["dt"]
    df["volume"] = df.groupby("breath_id")["area"].cumsum()

    # --- 2. Soft Physics Interactions (Equation of Motion) ---
    # Interaction terms derived from P ~ R*Flow + V/C
    df["R_u_in"] = df["R"] * df["u_in"]
    df["vol_C"] = df["volume"] / df["C"]

    # --- 3. Dynamics (Lags and Derivatives) ---
    # Add lags for u_in to capture system inertia
    # Groupby is used to ensure we don't shift data from the previous breath
    for lag in Config.LAG_STEPS:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # Add derivatives to capture trends
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff(1).fillna(0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in"].diff(2).fillna(0)

    # --- 4. Cleanup ---
    # Drop helper columns
    df = df.drop(columns=["area", "dt"])

    return df


def prepare_data(debug=Config.DEBUG, load_cached_data=True):
    """
    Main data processing pipeline.
    - Loads metadata CSVs.
    - Applies feature engineering.
    - Scales features (RobustScaler) while preserving binary u_out.
    - Reshapes to (N_breaths, 80, N_features).
    - Caches results to disk as .npy files.
    """
    seed_everything()

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        Config.CACHE_TEST_X,
        Config.CACHE_TEST_IDS,
        Config.CACHE_SCALER_CENTER,
        Config.CACHE_SCALER_SCALE,
    ]

    # --- Cache Loading ---
    if load_cached_data and all(os.path.exists(f) for f in cache_files):
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        train_x = np.load(Config.CACHE_TRAIN_X)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        val_x = np.load(Config.CACHE_VAL_X)
        val_y = np.load(Config.CACHE_VAL_Y)
        test_x = np.load(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS)
        return train_x, train_y, val_x, val_y, test_x, test_ids

    print("Cache not found or disabled. Processing data from scratch...")

    # --- Data Loading ---
    print("Loading metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Sample a subset of breaths
    if debug:
        print("DEBUG MODE: Sampling subset of data...")
        train_breaths = train_df["breath_id"].unique()[:100]
        val_breaths = val_df["breath_id"].unique()[:50]
        test_breaths = test_df["breath_id"].unique()[:50]

        train_df = train_df[train_df["breath_id"].isin(train_breaths)].copy()
        val_df = val_df[val_df["breath_id"].isin(val_breaths)].copy()
        test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

    # Store test_ids for final submission mapping
    test_ids = test_df["id"].values

    # --- Feature Engineering ---
    print("Applying Physics-Fidelity feature engineering...")
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    # Identify columns
    exclude_cols = ["id", "breath_id", "pressure"]
    all_feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Separate u_out (binary) from continuous features to avoid scaling it
    continuous_cols = [c for c in all_feature_cols if c != "u_out"]

    print(f"Continuous features ({len(continuous_cols)}): {continuous_cols}")
    print("Binary feature preserved: u_out")

    # Extract Targets
    train_y = train_df["pressure"].values
    val_y = val_df["pressure"].values

    # --- Scaling ---
    print("Fitting RobustScaler on training data...")
    scaler = RobustScaler()

    # Extract continuous parts
    train_cont = train_df[continuous_cols].values.astype(np.float32)
    val_cont = val_df[continuous_cols].values.astype(np.float32)
    test_cont = test_df[continuous_cols].values.astype(np.float32)

    # Fit and Transform
    scaler.fit(train_cont)
    train_cont = scaler.transform(train_cont)
    val_cont = scaler.transform(val_cont)
    test_cont = scaler.transform(test_cont)

    # Concatenate u_out back (as the last column)
    # This ensures u_out remains strictly 0.0 or 1.0
    train_u_out = train_df[["u_out"]].values.astype(np.float32)
    val_u_out = val_df[["u_out"]].values.astype(np.float32)
    test_u_out = test_df[["u_out"]].values.astype(np.float32)

    train_x_flat = np.concatenate([train_cont, train_u_out], axis=1)
    val_x_flat = np.concatenate([val_cont, val_u_out], axis=1)
    test_x_flat = np.concatenate([test_cont, test_u_out], axis=1)

    # --- Reshaping ---
    # Reshape to (N_breaths, 80, N_features)
    num_features = train_x_flat.shape[1]

    train_x = train_x_flat.reshape(-1, Config.SEQ_LEN, num_features)
    val_x = val_x_flat.reshape(-1, Config.SEQ_LEN, num_features)
    test_x = test_x_flat.reshape(-1, Config.SEQ_LEN, num_features)

    train_y = train_y.reshape(-1, Config.SEQ_LEN)
    val_y = val_y.reshape(-1, Config.SEQ_LEN)

    print(f"Processed Train Shape: {train_x.shape}")
    print(f"Processed Val Shape:   {val_x.shape}")

    # --- Caching ---
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    np.save(Config.CACHE_TRAIN_X, train_x)
    np.save(Config.CACHE_TRAIN_Y, train_y)
    np.save(Config.CACHE_VAL_X, val_x)
    np.save(Config.CACHE_VAL_Y, val_y)
    np.save(Config.CACHE_TEST_X, test_x)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    # Save scaler attributes manually to avoid pickle
    np.save(Config.CACHE_SCALER_CENTER, scaler.center_)
    np.save(Config.CACHE_SCALER_SCALE, scaler.scale_)

    return train_x, train_y, val_x, val_y, test_x, test_ids
