import os
import pandas as pd
import numpy as np
from library.config import Config


def impute_ground(df):
    """
    Implements geometric consistency for ground contact.
    When nfl_player_id_2 is 'G', sets Player 2's kinematics to be identical to Player 1's.
    Sets 'is_ground' flag.
    """
    # Create is_ground flag
    # nfl_player_id_2 is object/string in metadata
    df["is_ground"] = (df["nfl_player_id_2"] == "G").astype("int8")

    # List of kinematic columns to copy from 1 to 2
    cols_to_fix = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Identify ground rows
    ground_mask = df["is_ground"] == 1

    for col in cols_to_fix:
        col_1 = f"{col}_1"
        col_2 = f"{col}_2"

        # Ensure columns exist before attempting copy
        if col_1 in df.columns and col_2 in df.columns:
            # Impute P2 values with P1 values where P2 is Ground
            df.loc[ground_mask, col_2] = df.loc[ground_mask, col_1]

    return df


def add_kinematics(df):
    """
    Calculates distance, log_distance, and closing_speed.
    """
    # 1. Euclidean Distance
    df["dx"] = df["x_position_1"] - df["x_position_2"]
    df["dy"] = df["y_position_1"] - df["y_position_2"]
    df["distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    # 2. Log Distance (for resolution near 0)
    df["log_distance"] = np.log1p(df["distance"])

    # 3. Closing Speed
    # Convert direction (deg) and speed to velocity components
    # Assumption: 0 degrees is Y-axis, increasing clockwise (NFL standard)
    def get_velocity(speed_col, dir_col):
        # Fill NaNs with 0 to avoid errors
        s = df[speed_col].fillna(0)
        d = df[dir_col].fillna(0)
        theta = np.radians(d)
        vx = s * np.sin(theta)
        vy = s * np.cos(theta)
        return vx, vy

    vx1, vy1 = get_velocity("speed_1", "direction_1")
    vx2, vy2 = get_velocity("speed_2", "direction_2")

    # Relative velocity (P1 - P2)
    dvx = vx1 - vx2
    dvy = vy1 - vy2

    # Projection of relative velocity onto relative position vector
    # Closing speed is positive when getting closer
    dot_product = df["dx"] * dvx + df["dy"] * dvy

    # Clamped denominator to prevent division by zero
    denom = df["distance"].clip(lower=1e-6)

    # Closing speed: Positive means closing in
    # Formula: - (r . v) / |r|
    df["closing_speed"] = -(dot_product / denom)

    return df


def create_wide_features(df):
    """
    Constructs the temporal window features using vectorized shifts.
    """
    # Define the features to lag
    features = Config.FEATURES_PER_STEP
    half_window = Config.HALF_WINDOW_SIZE

    # Sort to ensure temporal order
    # Grouping keys: game_play, nfl_player_id_1, nfl_player_id_2
    # We include step to sort chronologically
    sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Create a unique group ID for boundary checking
    df["group_id"] = (
        df["game_play"].astype(str)
        + "_"
        + df["nfl_player_id_1"].astype(str)
        + "_"
        + df["nfl_player_id_2"].astype(str)
    )

    # List to store feature dataframes
    shifted_dfs = []

    # Lags: from -half_window to +half_window
    # e.g., -5, ..., 0, ..., 5
    shifts = range(-half_window, half_window + 1)

    for i in shifts:
        # shift_val: The argument to pd.shift().
        # If i = -5 (we want t-5), we need to look at the row 5 steps 'up'.
        # df.shift(5) moves the row 5 steps up to the current position.
        # So shift_val = -i.
        shift_val = -i

        # Shift features and validation columns
        shifted_data = df[features].shift(shift_val)
        shifted_group = df["group_id"].shift(shift_val)
        shifted_step = df["step"].shift(shift_val)

        # Validation Mask:
        # 1. Must belong to the same pair (group_id)
        # 2. Must be exactly 'i' steps away (handles missing rows/gaps)
        valid_mask = (shifted_group == df["group_id"]) & (
            shifted_step == (df["step"] + i)
        )

        # Apply mask: Invalid shifts become 0 (padding)
        # We fill NaNs created by shift with 0 as well
        shifted_data = shifted_data.where(valid_mask, 0)
        shifted_data = shifted_data.fillna(0)

        # Rename columns to flatten the window
        # Format: feature_step{idx} where idx is 0 to window_size-1
        # idx 0 corresponds to t - half_window
        step_idx = i + half_window
        shifted_data.columns = [f"{col}_step{step_idx}" for col in features]

        shifted_dfs.append(shifted_data)

    # Concatenate all shifted features horizontally
    wide_df = pd.concat(shifted_dfs, axis=1)

    # Combine with original metadata columns
    # We keep the identifiers and target
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
    ]
    # Filter for columns that actually exist (e.g. 'contact' might be in train but not raw test if not merged)
    # But here we assume df came from metadata which has these cols
    available_meta = [c for c in meta_cols if c in df.columns]

    result = pd.concat([df[available_meta], wide_df], axis=1)

    return result


def process_data(split, load_cached_data=True):
    """
    Main processing function.
    Args:
        split (str): 'train', 'validation', or 'test'
        load_cached_data (bool): Whether to try loading from cache
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data from scratch...")

    # 1. Load Metadata
    meta_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
    df_meta = pd.read_csv(meta_path)

    # 2. Load Tracking
    # Validation set uses training tracking data
    track_split = "test" if split == "test" else "train"
    track_path = os.path.join(Config.INPUT_DIR, f"{track_split}_player_tracking.csv")
    df_track = pd.read_csv(track_path)

    # Filter tracking to relevant game_plays to optimize memory
    relevant_plays = df_meta["game_play"].unique()
    df_track = df_track[df_track["game_play"].isin(relevant_plays)].copy()

    # 3. Merge Tracking Data
    # Prepare P2 ID for merging (handle 'G' which is string)
    # We create a numeric join key, 'G' becomes NaN
    df_meta["nfl_player_id_2_merge"] = pd.to_numeric(
        df_meta["nfl_player_id_2"], errors="coerce"
    )

    # Select specific tracking columns
    track_cols = ["game_play", "step", "nfl_player_id"] + Config.TRACKING_COLS

    # Merge Player 1
    df_merged = df_meta.merge(
        df_track[track_cols].add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
    )

    # Merge Player 2
    df_merged = df_merged.merge(
        df_track[track_cols].add_suffix("_2"),
        left_on=["game_play", "step", "nfl_player_id_2_merge"],
        right_on=["game_play_2", "step_2", "nfl_player_id_2"],
        how="left",
        suffixes=("", "_tracking"),
    )

    # Cleanup merge columns
    drop_cols = [
        "game_play_1",
        "step_1",
        "nfl_player_id_1_1",
        "game_play_2",
        "step_2",
        "nfl_player_id_2_2",
        "nfl_player_id_2_merge",
    ]
    df_merged.drop(
        columns=[c for c in drop_cols if c in df_merged.columns], inplace=True
    )

    # 4. Impute Ground
    df_merged = impute_ground(df_merged)

    # 5. Add Kinematics
    df_merged = add_kinematics(df_merged)

    # 6. Create Wide Features (Temporal Window)
    df_final = create_wide_features(df_merged)

    # 7. Save to Cache
    print(f"Saving {split} features to {cache_path}...")
    df_final.to_parquet(cache_path, index=False)

    return df_final
