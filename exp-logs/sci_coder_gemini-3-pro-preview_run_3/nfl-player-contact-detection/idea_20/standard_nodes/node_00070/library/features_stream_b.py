import pandas as pd
import numpy as np
import os
import library.config as C
import library.utils as U


def generate_stream_b_features(df_stream_b, mode="train", load_cached_data=True):
    """
    Generates features for Stream B (Player-Ground Impact).
    Focuses on Finite-Difference Ego-Dynamics and raw kinematics.
    Excludes visual and relational features.

    Args:
        df_stream_b (pd.DataFrame): Merged tracking data for player-ground pairs.
        mode (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): Feature matrix.
            y (np.array): Target vector.
            ids (np.array): Contact IDs.
    """
    # 1. Caching Setup
    cache_dir = C.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_X = os.path.join(cache_dir, f"features_stream_b_{mode}_X.parquet")
    cache_path_y = os.path.join(cache_dir, f"features_stream_b_{mode}_y.npy")
    cache_path_ids = os.path.join(cache_dir, f"features_stream_b_{mode}_ids.npy")

    if load_cached_data:
        if (
            os.path.exists(cache_path_X)
            and os.path.exists(cache_path_y)
            and os.path.exists(cache_path_ids)
        ):
            print(f"Loading Stream B features from cache for {mode}...")
            X = pd.read_parquet(cache_path_X)
            y = np.load(cache_path_y)
            ids = np.load(cache_path_ids, allow_pickle=True)
            return X, y, ids
        else:
            print(f"Cache miss for Stream B ({mode}). Generating features...")

    print(f"Generating Stream B features for {len(df_stream_b)} rows...")

    # Working copy
    df = df_stream_b.copy()

    # 2. Column Mapping (P1 -> Base)
    # Since P2 is Ground ('G'), we only care about P1 kinematics.
    # We map _p1 columns to the base names expected by the config.
    rename_map = {
        "x_position_p1": "x_position",
        "y_position_p1": "y_position",
        "speed_p1": "speed",
        "acceleration_p1": "acceleration",
        "orientation_p1": "orientation",
        "direction_p1": "direction",
    }
    df = df.rename(columns=rename_map)

    # 3. Base Kinematic Features (Sin/Cos)
    # Convert degrees to radians and compute components
    for col in ["orientation", "direction"]:
        rad = np.deg2rad(df[col].fillna(0))
        df[f"{col}_sin"] = np.sin(rad).astype(np.float32)
        df[f"{col}_cos"] = np.cos(rad).astype(np.float32)

    # 4. Finite-Difference Ego-Dynamics
    print("Computing Finite-Difference Ego-Dynamics...")

    # Calculate relative angle between motion (direction) and body facing (orientation)
    # angle_diff = direction - orientation
    # We use the radians calculated above.
    dir_rad = np.deg2rad(df["direction"].fillna(0))
    orient_rad = np.deg2rad(df["orientation"].fillna(0))

    # We need to handle the case where speed is 0 or direction is NaN
    # If speed is 0, surge/sway are 0.

    # Project Velocity onto Body Frame
    # Surge: Component parallel to orientation (Cos)
    # Sway: Component perpendicular to orientation (Sin)
    # Note: Standard rotation logic.
    # v_surge = |v| * cos(dir - orient)
    # v_sway = |v| * sin(dir - orient)

    angle_diff = dir_rad - orient_rad
    df["v_surge"] = (df["speed"] * np.cos(angle_diff)).astype(np.float32)
    df["v_sway"] = (df["speed"] * np.sin(angle_diff)).astype(np.float32)

    # Prepare for temporal operations
    # Group by Play and Player (P1) to ensure boundaries
    # Note: contact_id is unique per step, so we need to group by entity over time
    df["entity_id"] = df["game_play"] + "_" + df["nfl_player_id_1"].astype(str)

    # Sort is critical for diff/shift
    df = df.sort_values(["game_play", "entity_id", "step"]).reset_index(drop=True)

    grouper = df.groupby(["game_play", "entity_id"])

    # Calculate Ego-Acceleration (Derivative of Projected Velocity)
    # a_surge(t) = v_surge(t) - v_surge(t-1)
    # a_sway(t) = v_sway(t) - v_sway(t-1)
    # Using shift(1) to get t-1
    df["a_surge"] = (
        (df["v_surge"] - grouper["v_surge"].shift(1)).fillna(0).astype(np.float32)
    )
    df["a_sway"] = (
        (df["v_sway"] - grouper["v_sway"].shift(1)).fillna(0).astype(np.float32)
    )

    # Sensor Kinematics (Cite 00027, 00034)
    # Jerk: Derivative of Acceleration
    df["jerk"] = (
        (df["acceleration"] - grouper["acceleration"].shift(1))
        .fillna(0)
        .astype(np.float32)
    )

    # Angular Velocity: Derivative of Orientation
    # Handle wrap-around (0 vs 360)
    o_curr = df["orientation"].fillna(0)
    o_prev = grouper["orientation"].shift(1).fillna(0)
    diff = o_curr - o_prev
    # Normalize to [-180, 180]
    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)
    df["angular_velocity"] = np.abs(diff).astype(np.float32)

    # 5. Temporal Pyramids (Lags)
    print("Applying temporal pyramids (lags)...")

    base_features = C.STREAM_B_BASE_FEATURES
    lags = C.LAG_OFFSETS

    # Re-instantiate grouper in case df order changed (it shouldn't have, but safety)
    grouper = df.groupby(["game_play", "entity_id"])

    for feature in base_features:
        if feature not in df.columns:
            # Handle cases where feature might be missing (e.g. if config changed)
            # For Stream B, we expect all base features to be derived above
            print(f"Warning: Base feature {feature} missing. Filling 0.")
            df[feature] = 0.0

        for lag in lags:
            if lag == 0:
                continue

            # Future Lag (t + k) -> shift(-k)
            col_name_future = f"{feature}_lag_{lag}"
            df[col_name_future] = (
                grouper[feature].shift(-lag).fillna(0).astype(np.float32)
            )

            # Past Lag (t - k) -> shift(k)
            col_name_past = f"{feature}_lag_minus_{lag}"
            df[col_name_past] = grouper[feature].shift(lag).fillna(0).astype(np.float32)

    # 6. Final Selection & Cleaning
    expected_cols = C.STREAM_B_COLS

    # Check for missing columns
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} expected columns. Filling with 0.")
        for c in missing_cols:
            df[c] = 0.0

    # Select X
    X = df[expected_cols].copy()

    # Select y (target)
    if "contact" in df.columns:
        y = df["contact"].values.astype(np.int8)
    else:
        y = np.zeros(len(df), dtype=np.int8)

    # Select IDs
    if "contact_id" in df.columns:
        ids = df["contact_id"].values
    else:
        # Reconstruct: game_play_step_p1_G
        ids = (
            df["game_play"]
            + "_"
            + df["step"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_G"
        ).values

    # Memory Optimization
    X = U.reduce_mem_usage(X, verbose=False)

    # 7. Save to Cache
    print(f"Saving Stream B features to {cache_dir}...")
    X.to_parquet(cache_path_X, index=False)
    np.save(cache_path_y, y)
    np.save(cache_path_ids, ids)

    return X, y, ids
