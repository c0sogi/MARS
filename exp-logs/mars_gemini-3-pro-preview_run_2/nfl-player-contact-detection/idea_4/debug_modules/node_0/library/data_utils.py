import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import library.config as config


def _load_tracking_data(path, game_plays=None):
    """
    Loads tracking data and filters for relevant game_plays to save memory.
    """
    # Read only necessary columns to save memory
    df = pd.read_csv(path, usecols=config.TRACKING_READ_COLS)

    if game_plays is not None:
        df = df[df["game_play"].isin(game_plays)].copy()

    return df


def _calculate_velocity_components(df, suffix):
    """
    Converts speed and direction into vx and vy components.
    Assumes NFL coordinate system: 0 degrees is North (Y-axis), 90 is East (X-axis).
    vx = speed * sin(deg2rad(direction))
    vy = speed * cos(deg2rad(direction))
    """
    # Convert direction to radians
    rads = np.deg2rad(df[f"direction{suffix}"])

    # Calculate components
    df[f"vx{suffix}"] = df[f"speed{suffix}"] * np.sin(rads)
    df[f"vy{suffix}"] = df[f"speed{suffix}"] * np.cos(rads)
    return df


def _engineer_interaction_features(df):
    """
    Calculates distance, log_distance, and closing_speed.
    """
    # 1. Distance
    df["dx"] = df["x_position_1"] - df["x_position_2"]
    df["dy"] = df["y_position_1"] - df["y_position_2"]
    df["distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    # 2. Log Distance (for better resolution near 0)
    df["log_distance"] = np.log1p(df["distance"])

    # 3. Closing Speed
    # Calculate velocity components for both players
    df = _calculate_velocity_components(df, "_1")
    df = _calculate_velocity_components(df, "_2")

    # Relative velocity (v1 - v2)
    dvx = df["vx_1"] - df["vx_2"]
    dvy = df["vy_1"] - df["vy_2"]

    # Closing speed = - (v_rel . r_rel) / |r|
    # Dot product of relative velocity and relative position
    dot_prod = dvx * df["dx"] + dvy * df["dy"]

    # Clamp distance to avoid division by zero
    clamped_dist = df["distance"].clip(lower=1e-6)

    # Closing speed (positive means getting closer)
    df["closing_speed"] = -(dot_prod / clamped_dist)

    # 4. Is Ground
    # This should already be handled during merge, but ensuring it exists
    if "is_ground" not in df.columns:
        # If nfl_player_id_2 is 'G', it's ground.
        # Note: After merge, 'G' might be lost if we converted to numeric,
        # but we handle 'G' logic in merge.
        # We assume the caller handles the creation of 'is_ground' or we do it here if columns exist.
        pass

    return df


def _create_wide_features(df, feature_cols, window_size):
    """
    Creates lag features using vectorized shift.
    Flattens the temporal window into columns.
    """
    # Ensure data is sorted for correct shifting
    # We group by the specific contact pair to ensure we don't shift across different pairs
    sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
    df = df.sort_values(sort_cols)

    # Group by pair
    grouper = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

    lagged_dfs = []

    # Window is centered: e.g., size 11 -> -5 to +5
    half_window = window_size // 2
    offsets = range(-half_window, half_window + 1)

    for offset in offsets:
        # shift(k) shifts data down by k.
        # We want data from t+offset at row t.
        # If offset is -5 (past), we want the value from 5 rows ago. shift(5).
        # If offset is +5 (future), we want the value from 5 rows ahead. shift(-5).
        # So we use shift(-offset).

        # Select features and rename
        shifted = grouper[feature_cols].shift(-offset)
        shifted.columns = [f"{col}_lag_{offset}" for col in feature_cols]
        lagged_dfs.append(shifted)

    # Concatenate all lag features horizontally
    df_lags = pd.concat(lagged_dfs, axis=1)

    # Join back to original dataframe
    # Since indices align (we just shifted), we can concat horizontally
    # However, concat with existing df might duplicate columns if we are not careful.
    # We only want the lagged columns and the identifiers/targets from the original df.

    meta_cols = [
        c for c in df.columns if c not in feature_cols and c not in df_lags.columns
    ]
    # We keep the original metadata columns
    df_final = pd.concat([df[meta_cols], df_lags], axis=1)

    return df_final


def load_and_process_data(split="train", load_cached_data=True):
    """
    Main function to load, process, and cache data for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with wide features.
    """
    # Determine paths
    if split == "train":
        meta_path = config.TRAIN_METADATA_PATH
        tracking_path = config.TRAIN_TRACKING_PATH
        cache_path = config.TRAIN_FEATURES_PATH
    elif split == "val":
        meta_path = config.VAL_METADATA_PATH
        tracking_path = config.TRAIN_TRACKING_PATH  # Val uses train tracking file
        cache_path = config.VAL_FEATURES_PATH
    elif split == "test":
        meta_path = config.TEST_METADATA_PATH
        tracking_path = config.TEST_TRACKING_PATH
        cache_path = config.TEST_FEATURES_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data from scratch...")

    # 2. Load Metadata
    df_meta = pd.read_csv(meta_path)

    # 3. Load Tracking
    # Optimization: Only load tracking for game_plays in this split
    unique_gps = df_meta["game_play"].unique()
    df_tracking = _load_tracking_data(tracking_path, unique_gps)

    # 4. Merge Tracking Data

    # Prepare Player 1 Merge
    # Tracking keys: game_play, step, nfl_player_id
    # Meta keys: game_play, step, nfl_player_id_1

    # Ensure types match for merge
    df_meta["nfl_player_id_1"] = pd.to_numeric(
        df_meta["nfl_player_id_1"], errors="coerce"
    )

    # Merge P1
    df_merged = df_meta.merge(
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    ).drop(
        columns=["nfl_player_id"]
    )  # Drop redundant column

    # Rename P1 columns
    rename_dict_1 = {col: f"{col}_1" for col in config.PLAYER_TRACKING_FEATURES}
    df_merged = df_merged.rename(columns=rename_dict_1)

    # Prepare Player 2 Merge
    # Handle Ground: 'G' in nfl_player_id_2
    df_merged["is_ground"] = (df_merged["nfl_player_id_2"] == "G").astype(int)

    # Convert P2 ID to numeric, 'G' becomes NaN
    df_merged["nfl_player_id_2_num"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    # Merge P2
    df_merged = df_merged.merge(
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_2_track"),
    ).drop(columns=["nfl_player_id", "nfl_player_id_2_num"])

    # Rename P2 columns
    rename_dict_2 = {col: f"{col}_2" for col in config.PLAYER_TRACKING_FEATURES}
    df_merged = df_merged.rename(columns=rename_dict_2)

    # Fill missing P2 data (Ground or missing tracking) with 0
    # This is crucial for the network to handle 'Ground' as a zero-vector entity
    p2_cols = [f"{col}_2" for col in config.PLAYER_TRACKING_FEATURES]
    df_merged[p2_cols] = df_merged[p2_cols].fillna(0)

    # Also fill missing P1 data with 0 (rare but possible)
    p1_cols = [f"{col}_1" for col in config.PLAYER_TRACKING_FEATURES]
    df_merged[p1_cols] = df_merged[p1_cols].fillna(0)

    # 5. Feature Engineering (Interaction)
    df_merged = _engineer_interaction_features(df_merged)

    # 6. Create Wide Features (Lags)
    # Define features to lag based on config
    # We need to ensure these columns exist
    features_to_lag = config.FEATURES_TO_LAG

    # Check if all required features exist
    missing_feats = [f for f in features_to_lag if f not in df_merged.columns]
    if missing_feats:
        # Some might be intermediate like 'vx_1', but config lists base + interaction
        # If config lists 'vx_1' and we didn't keep it, we might have an issue.
        # Based on config provided in prompt:
        # FEATURES_TO_LAG includes x_position_1, ..., distance, log_distance, closing_speed, is_ground
        # We have created all of these.
        raise ValueError(f"Missing features before lagging: {missing_feats}")

    df_wide = _create_wide_features(df_merged, features_to_lag, config.WINDOW_SIZE)

    # 7. Final Cleanup
    # Fill NaNs generated by shifting (at start/end of plays) with 0
    lag_cols = [c for c in df_wide.columns if "_lag_" in c]
    df_wide[lag_cols] = df_wide[lag_cols].fillna(0)

    # Cache result
    print(f"Saving {split} data to cache: {cache_path}")
    df_wide.to_parquet(cache_path)

    return df_wide


def get_feature_columns(df):
    """
    Returns the list of feature columns (lagged columns).
    """
    return [c for c in df.columns if "_lag_" in c]


def scale_data(df_train, df_val=None, df_test=None):
    """
    Fits scaler on train, transforms train/val/test.
    Returns the feature matrices (numpy arrays) and the scaler.
    """
    feature_cols = get_feature_columns(df_train)

    print(f"Scaling {len(feature_cols)} features...")

    scaler = StandardScaler()

    # Fit on train
    X_train = scaler.fit_transform(df_train[feature_cols].values.astype(np.float32))

    X_val = None
    if df_val is not None:
        X_val = scaler.transform(df_val[feature_cols].values.astype(np.float32))

    X_test = None
    if df_test is not None:
        X_test = scaler.transform(df_test[feature_cols].values.astype(np.float32))

    return X_train, X_val, X_test, scaler
