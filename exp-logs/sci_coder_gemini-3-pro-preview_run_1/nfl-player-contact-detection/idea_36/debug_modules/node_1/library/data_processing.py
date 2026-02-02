import os
import numpy as np
import pandas as pd
from library import config, utils


def _load_tracking_data(path):
    """
    Loads and optimizes tracking data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tracking data not found at {path}")

    df = pd.read_csv(path)
    df = utils.reduce_mem_usage(df)

    # Standardize direction to radians (0 is North/Y, increasing clockwise)
    # v_x = speed * sin(theta), v_y = speed * cos(theta)
    # We'll just convert to radians here for later use
    if "direction" in df.columns:
        df["direction_rad"] = np.deg2rad(df["direction"].fillna(0))

    # Select only necessary columns to save memory during merge
    cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "direction_rad",
        "acceleration",
        "sa",
    ]
    # Filter columns that exist
    cols = [c for c in cols if c in df.columns]

    return df[cols]


def _merge_tracking(df_meta, df_tracking, player_num):
    """
    Merges tracking data for a specific player (1 or 2).
    """
    suffix = f"_p{player_num}"
    player_col = f"nfl_player_id_{player_num}"

    # Ensure ID types match
    df_meta[player_col] = df_meta[player_col].astype(int)

    # Merge
    df_merged = pd.merge(
        df_meta,
        df_tracking,
        left_on=["game_play", "step", player_col],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename tracking columns with suffix
    track_cols = [
        c
        for c in df_tracking.columns
        if c not in ["game_play", "step", "nfl_player_id"]
    ]
    rename_map = {c: f"{c}{suffix}" for c in track_cols}
    df_merged = df_merged.rename(columns=rename_map)

    # Drop the redundant nfl_player_id column from tracking if it exists
    if "nfl_player_id" in df_merged.columns:
        df_merged = df_merged.drop(columns=["nfl_player_id"])

    return df_merged


def _apply_quadratic_gating(df):
    """
    Applies Relaxed Quadratic Reachability Gating.
    Models relative motion to find min distance in window [-1s, +1s].
    """
    # Constants
    WINDOW_SECONDS = 1.0

    # Extract kinematics
    x1, y1 = df["x_position_p1"], df["y_position_p1"]
    x2, y2 = df["x_position_p2"], df["y_position_p2"]
    s1, d1 = df["speed_p1"], df["direction_rad_p1"]
    s2, d2 = df["speed_p2"], df["direction_rad_p2"]

    # Calculate Velocity Components
    vx1 = s1 * np.sin(d1)
    vy1 = s1 * np.cos(d1)
    vx2 = s2 * np.sin(d2)
    vy2 = s2 * np.cos(d2)

    # Relative Position (r0) and Velocity (v_rel)
    rx = x1 - x2
    ry = y1 - y2
    rvx = vx1 - vx2
    rvy = vy1 - vy2

    # Current Distance
    current_dist_sq = rx**2 + ry**2
    df["distance"] = np.sqrt(current_dist_sq)

    # Quadratic Distance Squared: D^2(t) = |r0 + v*t|^2
    # D^2(t) = (rx + rvx*t)^2 + (ry + rvy*t)^2
    # D^2(t) = (rvx^2 + rvy^2)t^2 + 2(rx*rvx + ry*rvy)t + (rx^2 + ry^2)
    # Form: At^2 + Bt + C
    A = rvx**2 + rvy**2
    B = 2 * (rx * rvx + ry * rvy)
    C = current_dist_sq

    # Find time of minimum distance t* = -B / (2A)
    # Avoid division by zero
    A = np.where(A < 1e-6, 1e-6, A)
    t_star = -B / (2 * A)

    # Clamp t* to window
    t_clamped = np.clip(t_star, -WINDOW_SECONDS, WINDOW_SECONDS)

    # Calculate min squared distance at t_clamped
    min_dist_sq = A * (t_clamped**2) + B * t_clamped + C
    min_dist = np.sqrt(np.maximum(0, min_dist_sq))

    # Filter
    # We keep if min_dist < Threshold
    mask = min_dist < config.GATING_THRESHOLD

    # Also keep if current distance is already low (redundant but safe)
    mask = mask | (df["distance"] < config.GATING_THRESHOLD)

    # Debug stats
    kept_ratio = mask.mean()
    print(f"  Gating Survival Rate: {kept_ratio:.2%}")

    return df[mask].copy()


def _process_data_pipeline(
    metadata_path, tracking_path, is_train=True, debug=False, sample_size=None
):
    """
    Core pipeline: Load -> Split (G/P) -> Merge -> Gate -> Combine.
    """
    print(f"Processing {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    if debug and sample_size:
        print(f"  Debug: Sampling {sample_size} rows...")
        if len(df_meta) > sample_size:
            df_meta = df_meta.sample(
                n=sample_size, random_state=config.SEED
            ).reset_index(drop=True)

    df_track = _load_tracking_data(tracking_path)

    # 1. Split Player-Player vs Player-Ground
    # Ground is marked by 'G' in nfl_player_id_2
    mask_ground = df_meta["nfl_player_id_2"] == "G"
    df_ground = df_meta[mask_ground].copy()
    df_players = df_meta[~mask_ground].copy()

    print(f"  Split: {len(df_players)} Player-Player, {len(df_ground)} Player-Ground")

    # 2. Process Player-Player
    if not df_players.empty:
        # Merge P1
        df_players = _merge_tracking(df_players, df_track, 1)
        # Merge P2
        df_players = _merge_tracking(df_players, df_track, 2)

        # Drop rows where tracking is missing (cannot gate or predict)
        len_before = len(df_players)
        df_players = df_players.dropna(subset=["x_position_p1", "x_position_p2"])
        if len(df_players) < len_before:
            print(
                f"  Dropped {len_before - len(df_players)} rows due to missing tracking data."
            )

        # Apply Gating
        df_players = _apply_quadratic_gating(df_players)

    # 3. Process Player-Ground
    if not df_ground.empty:
        # Merge P1 only
        df_ground = _merge_tracking(df_ground, df_track, 1)

        # Set Sentinel Values
        df_ground["distance"] = -1.0

        # Fill P2 columns with 0 or NaN to match schema
        # We need to ensure columns exist so concat works
        if not df_players.empty:
            p2_cols = [c for c in df_players.columns if "_p2" in c]
            for col in p2_cols:
                df_ground[col] = 0.0

    # 4. Combine
    df_final = pd.concat([df_players, df_ground], axis=0, ignore_index=True)

    # Explicitly cast nfl_player_id_2 to string to handle mixed types (int vs 'G') for Parquet
    df_final["nfl_player_id_2"] = df_final["nfl_player_id_2"].astype(str)

    # Ensure consistent schema
    df_final = utils.reduce_mem_usage(df_final)

    print(f"  Final Dataset Size: {len(df_final)}")
    return df_final


@utils.cache_result(file_format="parquet")
def process_train_data(debug=False, sample_size=10000, load_cached_data=True):
    """
    Processes training data with gating and sentinel logic.
    """
    return _process_data_pipeline(
        config.TRAIN_METADATA_PATH,
        config.TRAIN_TRACKING_PATH,
        is_train=True,
        debug=debug,
        sample_size=sample_size,
    )


@utils.cache_result(file_format="parquet")
def process_val_data(debug=False, sample_size=10000, load_cached_data=True):
    """
    Processes validation data.
    """
    return _process_data_pipeline(
        config.VAL_METADATA_PATH,
        config.TRAIN_TRACKING_PATH,  # Val uses train tracking file
        is_train=True,
        debug=debug,
        sample_size=sample_size,
    )


@utils.cache_result(file_format="parquet")
def process_test_data(load_cached_data=True):
    """
    Processes test data.
    Note: Gating is technically optional for test, but good for efficiency.
    We apply it to be consistent, but maybe with a safer threshold?
    Strategy says: "Run Scouts on the Entire Gated Survivor Pool".
    For Inference, we should probably be careful about filtering.
    However, 3.0 yards is very safe. We will apply it to save inference time.
    """
    return _process_data_pipeline(
        config.TEST_METADATA_PATH,
        config.TEST_TRACKING_PATH,
        is_train=False,
        debug=False,
    )
