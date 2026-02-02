import os
import gc
import numpy as np
import pandas as pd
from library import config


def generate_features(split: str, load_cached_data: bool = True, debug: bool = False):
    """
    Main function to generate or load features for a specific data split.

    Args:
        split: 'train', 'validation', or 'test'
        load_cached_data: If True, try to load from disk first.
        debug: If True, process a smaller subset for debugging.

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        ids (pd.Series): Contact IDs
    """
    # Determine paths based on split
    if split == "train":
        meta_path = config.TRAIN_META_PATH
        tracking_path = config.TRAIN_TRACKING_PATH
        cache_path = config.TRAIN_CACHE_PATH
    elif split == "validation":
        meta_path = config.VAL_META_PATH
        tracking_path = config.TRAIN_TRACKING_PATH  # Validation comes from train source
        cache_path = config.VAL_CACHE_PATH
    elif split == "test":
        meta_path = config.TEST_META_PATH
        tracking_path = config.TEST_TRACKING_PATH
        cache_path = config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Caching Check
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        df = pd.read_parquet(cache_path)

        # Separate into X, y, ids
        # Assuming 'contact' is target and 'contact_id' is ID
        y = df["contact"]
        ids = df["contact_id"]
        X = df.drop(columns=["contact", "contact_id"])
        return X, y, ids

    print(f"Generating features for {split} split...")

    # 2. Load Data
    df_meta = pd.read_csv(meta_path)
    if debug:
        df_meta = df_meta.head(5000)

    # Load tracking data
    # We only need tracking for the game_plays present in the metadata
    relevant_games = df_meta["game_play"].unique()

    # Reading tracking data can be slow, so we use pyarrow engine if available or just standard
    # We filter immediately to save memory
    df_tracking = pd.read_csv(tracking_path)
    df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_games)].copy()

    # 3. Preprocess Tracking (Windowing)
    # We create a wide dataframe with lagged features
    df_tracking_wide = _create_windowed_tracking(df_tracking)

    # Free memory
    del df_tracking
    gc.collect()

    # 4. Merge and Feature Engineering
    df_features = _merge_and_compute_kinematics(df_meta, df_tracking_wide)

    # Free memory
    del df_tracking_wide
    gc.collect()

    # 5. Normalization
    # Define continuous columns (exclude ID, target, and binary flags)
    exclude_cols = [
        "contact_id",
        "contact",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "nfl_player_id_2_num",
        "datetime",
        "is_ground",
    ]
    # Also exclude path columns if they exist
    exclude_cols += [c for c in df_features.columns if "path_" in c]

    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    scaler_mean_path = os.path.join(config.WORKING_DIR, "scaler_mean.npy")
    scaler_scale_path = os.path.join(config.WORKING_DIR, "scaler_scale.npy")

    if split == "train":
        # Compute and save scaler stats
        print("Computing scaler statistics on training data...")
        means = df_features[feature_cols].mean().values
        stds = df_features[feature_cols].std().values

        # Avoid division by zero
        stds = np.where(stds == 0, 1.0, stds)

        np.save(scaler_mean_path, means)
        np.save(scaler_scale_path, stds)
    else:
        # Load scaler stats
        if not os.path.exists(scaler_mean_path):
            raise FileNotFoundError(
                "Scaler statistics not found. Run 'train' split first."
            )
        print("Loading scaler statistics...")
        means = np.load(scaler_mean_path)
        stds = np.load(scaler_scale_path)

    # Apply Normalization
    print("Applying normalization...")
    # Convert to float32 to save memory
    df_features[feature_cols] = ((df_features[feature_cols] - means) / stds).astype(
        np.float32
    )

    # 6. Save to Cache
    # We save the full dataframe including targets and IDs
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Select final columns to save
    # We keep contact_id and contact for reconstruction
    final_cols = ["contact_id", "contact"] + feature_cols + ["is_ground"]

    # Ensure 'contact' exists (it does in meta, but test might be placeholder)
    if "contact" not in df_features.columns:
        df_features["contact"] = 0  # Placeholder for test if missing

    df_save = df_features[final_cols]
    print(f"Saving features to {cache_path}...")
    df_save.to_parquet(cache_path, index=False)

    y = df_save["contact"]
    ids = df_save["contact_id"]
    X = df_save.drop(columns=["contact", "contact_id"])

    return X, y, ids


def _create_windowed_tracking(df_tracking):
    """
    Creates lagged features for tracking data.
    """
    print("Creating temporal window features...")

    # Sort to ensure shift works correctly
    df_tracking = df_tracking.sort_values(["game_play", "nfl_player_id", "step"])

    # Define columns to lag
    cols_to_lag = config.RAW_TRACKING_COLS

    # Base dataframe with keys
    df_base = df_tracking[["game_play", "nfl_player_id", "step"]].copy()

    # We will collect all lagged frames here
    lagged_dfs = []

    # Range of lags: e.g., -5 to +5
    lags = range(-config.HALF_WINDOW, config.HALF_WINDOW + 1)

    for lag in lags:
        # shift(lag): positive lag gets previous value (t-lag), negative lag gets future value
        # We want lag k to represent t+k.
        # If k=-5 (past), we want value from 5 steps ago. shift(5).
        # If k=5 (future), we want value from 5 steps ahead. shift(-5).
        # Let's stick to standard notation: lag k means t+k.
        # To get value at t+k into row t, we need shift(-k).

        shifted = df_tracking.groupby(["game_play", "nfl_player_id"])[
            cols_to_lag
        ].shift(-lag)

        # Rename columns
        shifted.columns = [f"{col}_lag_{lag}" for col in cols_to_lag]

        # We assume the index matches df_base because groupby.shift preserves index
        lagged_dfs.append(shifted)

    # Concatenate all lagged features
    df_wide = pd.concat([df_base] + lagged_dfs, axis=1)

    return df_wide


def _merge_and_compute_kinematics(df_meta, df_tracking_wide):
    """
    Merges tracking data to contact pairs and computes kinematic features.
    """
    print("Merging tracking data and computing kinematics...")

    # Prepare metadata
    # Ensure player IDs are consistent types
    df_meta["nfl_player_id_1"] = pd.to_numeric(
        df_meta["nfl_player_id_1"], errors="coerce"
    )

    # Handle 'G' in player 2
    df_meta["is_ground"] = (df_meta["nfl_player_id_2"] == "G").astype(int)
    df_meta["nfl_player_id_2_num"] = pd.to_numeric(
        df_meta["nfl_player_id_2"], errors="coerce"
    )

    # Merge Player 1
    df_merged = df_meta.merge(
        df_tracking_wide,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(
        columns=["nfl_player_id"]
    )  # Drop redundant join key

    # Rename P1 columns
    lag_cols = [c for c in df_tracking_wide.columns if "_lag_" in c]
    rename_map_p1 = {c: f"p1_{c}" for c in lag_cols}
    df_merged = df_merged.rename(columns=rename_map_p1)

    # Merge Player 2
    # We join on the numeric ID. 'G' will result in NaNs.
    df_merged = df_merged.merge(
        df_tracking_wide,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_p2"),
    ).drop(columns=["nfl_player_id"])

    # Rename P2 columns
    rename_map_p2 = {c: f"p2_{c}" for c in lag_cols}
    df_merged = df_merged.rename(columns=rename_map_p2)

    # --- Imputation & Handling Ground ---

    # Identify Ground rows
    is_ground = df_merged["is_ground"] == 1

    lags = range(-config.HALF_WINDOW, config.HALF_WINDOW + 1)

    # For each lag, handle Ground logic and Compute Interactions
    for lag in lags:
        # Base column names for this lag
        p1_x = f"p1_x_position_lag_{lag}"
        p1_y = f"p1_y_position_lag_{lag}"
        p1_s = f"p1_speed_lag_{lag}"
        p1_a = f"p1_acceleration_lag_{lag}"

        p2_x = f"p2_x_position_lag_{lag}"
        p2_y = f"p2_y_position_lag_{lag}"
        p2_s = f"p2_speed_lag_{lag}"
        p2_a = f"p2_acceleration_lag_{lag}"

        # 1. Handle Ground (Impute P2)
        # If Ground: P2 pos = P1 pos, P2 speed/accel = 0
        df_merged.loc[is_ground, p2_x] = df_merged.loc[is_ground, p1_x]
        df_merged.loc[is_ground, p2_y] = df_merged.loc[is_ground, p1_y]

        # For scalar kinematics, 0 is appropriate
        zero_cols = [p2_s, p2_a]
        # Also orientation/direction/sa if they exist
        for attr in ["orientation", "direction", "sa"]:
            col = f"p2_{attr}_lag_{lag}"
            if col in df_merged.columns:
                zero_cols.append(col)

        for col in zero_cols:
            df_merged.loc[is_ground, col] = 0.0

        # 2. Impute Missing Tracking (Non-Ground)
        # If P1 or P2 (non-ground) is missing, we fill with 0 or mean.
        # For simplicity and robustness in this script, we fill remaining NaNs with 0
        # (assuming missing tracking means out of play or stopped).
        # Note: A more complex imputer could use means, but we do that via scaler later?
        # No, scaler centers data. We need values now.
        # Let's forward fill or just fill 0. Given the physics features, 0 is safer than mean for distance calcs.
        # Actually, if P1 is missing, distance is undefined.
        # We fill with 0 to avoid crashes, but the model will likely learn to ignore.

        # Fill P1 NaNs
        p1_cols = [c for c in df_merged.columns if f"p1_" in c and f"_lag_{lag}" in c]
        df_merged[p1_cols] = df_merged[p1_cols].fillna(0.0)

        # Fill P2 NaNs
        p2_cols = [c for c in df_merged.columns if f"p2_" in c and f"_lag_{lag}" in c]
        df_merged[p2_cols] = df_merged[p2_cols].fillna(0.0)

        # 3. Compute Interactions
        # Distance
        dx = df_merged[p1_x] - df_merged[p2_x]
        dy = df_merged[p1_y] - df_merged[p2_y]
        df_merged[f"dist_lag_{lag}"] = np.sqrt(dx**2 + dy**2)

        # Relative Speed
        df_merged[f"rel_speed_lag_{lag}"] = np.abs(df_merged[p1_s] - df_merged[p2_s])

        # Relative Acceleration
        df_merged[f"rel_accel_lag_{lag}"] = np.abs(df_merged[p1_a] - df_merged[p2_a])

        # Orientation Diff (if available)
        p1_o = f"p1_orientation_lag_{lag}"
        p2_o = f"p2_orientation_lag_{lag}"
        if p1_o in df_merged.columns and p2_o in df_merged.columns:
            # Simple absolute difference, handling 360 wrap could be better but abs diff is a strong baseline
            # Correct angular difference: min(|a-b|, 360-|a-b|)
            diff = np.abs(df_merged[p1_o] - df_merged[p2_o])
            df_merged[f"rel_orient_lag_{lag}"] = np.minimum(diff, 360 - diff)

    return df_merged
