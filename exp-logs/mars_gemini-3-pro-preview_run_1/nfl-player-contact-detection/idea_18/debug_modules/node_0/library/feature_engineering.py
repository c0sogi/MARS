import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, save_cache_parquet, load_cache_parquet

logger = setup_logger("feature_engineering")


def compute_quadratic_min_dist(d0, v_rel, a_rel, lookahead=1.0):
    """
    Computes the minimum distance in the interval [0, lookahead]
    given the kinematic equation: d(t) = d0 + v_rel*t + 0.5*a_rel*t^2.
    """
    # Initialize min_dist with endpoints (t=0 and t=lookahead)
    d_end = d0 + v_rel * lookahead + 0.5 * a_rel * (lookahead**2)
    min_dist = np.minimum(d0, d_end)

    # Find critical points where derivative is 0: v_rel + a_rel*t = 0 -> t = -v_rel / a_rel
    # Mask for valid quadratic (a_rel != 0) to avoid division by zero
    nonzero_a = np.abs(a_rel) > 1e-6

    t_crit = np.zeros_like(d0)
    # Safe division
    t_crit[nonzero_a] = -v_rel[nonzero_a] / a_rel[nonzero_a]

    # Check if critical point is within valid time window (0, lookahead)
    valid_t = (t_crit > 0) & (t_crit < lookahead) & nonzero_a

    if np.any(valid_t):
        # Evaluate d(t) at critical point
        d_crit = (
            d0[valid_t]
            + v_rel[valid_t] * t_crit[valid_t]
            + 0.5 * a_rel[valid_t] * (t_crit[valid_t] ** 2)
        )

        # Distance is physically non-negative.
        # If the quadratic model crosses zero, the min distance is 0.
        d_crit = np.maximum(0, d_crit)

        # Update min_dist
        current_min = min_dist[valid_t]
        min_dist[valid_t] = np.minimum(current_min, d_crit)

    # Ensure final result is non-negative
    min_dist = np.maximum(0, min_dist)
    return min_dist


def compute_spectral_energy(shock_matrix):
    """
    Computes RMS of the shock component row-wise.
    shock_matrix: Array of shape (N_samples, Window_Size)
    """
    return np.sqrt(np.mean(np.square(shock_matrix), axis=1))


def process_dataset(
    metadata_path, tracking_path, dataset_type="train", load_cached_data=True
):
    """
    Main feature engineering pipeline.
    Loads data, merges tracking, computes kinematics, applies gating, and generates spectral features.
    """
    cache_file = f"features_{dataset_type}_full.parquet"

    # 1. Cache Check
    if load_cached_data:
        df_cached = load_cache_parquet(cache_file)
        if df_cached is not None:
            logger.info(f"Loaded {dataset_type} features from cache.")
            return df_cached

    logger.info(f"Processing {dataset_type} data from scratch...")

    # 2. Load Data
    df_meta = pd.read_csv(metadata_path)
    df_track = pd.read_csv(tracking_path)

    # Optimization: Downcast tracking types to save memory
    float_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
    ]
    for col in float_cols:
        if col in df_track.columns:
            df_track[col] = df_track[col].astype(np.float32)

    # 3. Merge Tracking Data
    logger.info("Merging tracking data...")

    # Prepare Player 1 Tracking
    df_track_p1 = df_track.copy()
    df_track_p1.columns = [
        f"{c}_p1" if c not in ["game_play", "step"] else c for c in df_track_p1.columns
    ]

    # Prepare Player 2 Tracking
    df_track_p2 = df_track.copy()
    df_track_p2.columns = [
        f"{c}_p2" if c not in ["game_play", "step"] else c for c in df_track_p2.columns
    ]

    # Ensure ID types match for merging
    df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
    df_track_p1["nfl_player_id_p1"] = df_track_p1["nfl_player_id_p1"].astype(str)

    # Merge P1
    df_merged = pd.merge(
        df_meta,
        df_track_p1,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id_p1"],
        how="left",
    )

    # Merge P2 (Handle Ground 'G')
    is_ground = df_merged["nfl_player_id_2"] == "G"

    # Create a clean merge key for P2
    df_merged["nfl_player_id_2_clean"] = df_merged["nfl_player_id_2"].astype(str)
    df_merged.loc[is_ground, "nfl_player_id_2_clean"] = "-1"  # Dummy ID for ground

    df_track_p2["nfl_player_id_p2"] = df_track_p2["nfl_player_id_p2"].astype(str)

    df_merged = pd.merge(
        df_merged,
        df_track_p2,
        left_on=["game_play", "step", "nfl_player_id_2_clean"],
        right_on=["game_play", "step", "nfl_player_id_p2"],
        how="left",
    )

    # Fill missing P2 tracking (Ground or missing data) with 0
    p2_cols = [c for c in df_merged.columns if c.endswith("_p2")]
    df_merged[p2_cols] = df_merged[p2_cols].fillna(0.0)

    # 4. Instantaneous Kinematics
    logger.info("Computing instantaneous kinematics...")

    x1, y1 = df_merged["x_position_p1"], df_merged["y_position_p1"]
    x2, y2 = df_merged["x_position_p2"], df_merged["y_position_p2"]
    s1, s2 = df_merged["speed_p1"], df_merged["speed_p2"]
    a1, a2 = df_merged["acceleration_p1"], df_merged["acceleration_p2"]

    # Distance
    df_merged["distance"] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # Sentinel for Ground
    df_merged.loc[is_ground, "distance"] = Config.FEATURES["SENTINEL_VALUE"]

    # Relative Kinematics (Scalar Invariants)
    df_merged["rel_speed"] = np.abs(s1 - s2)
    df_merged["rel_accel"] = np.abs(a1 - a2)

    # Kinetic Energy
    df_merged["ke_p1"] = 0.5 * (s1**2)
    df_merged["ke_p2"] = 0.5 * (s2**2)

    # 5. Quadratic Reachability Gating
    logger.info("Applying Quadratic Reachability Gating...")

    # Sort to compute temporal derivatives
    df_merged.sort_values(
        by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
    )

    # Identify groups
    group_id = (
        df_merged["game_play"]
        + "_"
        + df_merged["nfl_player_id_1"]
        + "_"
        + df_merged["nfl_player_id_2"].astype(str)
    )
    group_change = group_id != group_id.shift(1)

    # Estimate v_rel (rate of change of distance)
    d_prev = df_merged["distance"].shift(1)
    d_prev[group_change] = np.nan
    v_rel_est = (df_merged["distance"] - d_prev) / Config.GATING["TIME_STEP"]
    v_rel_est = v_rel_est.fillna(0.0)

    # Estimate a_rel (rate of change of v_rel)
    v_prev = v_rel_est.shift(1)
    v_prev[group_change] = np.nan
    a_rel_est = (v_rel_est - v_prev) / Config.GATING["TIME_STEP"]
    a_rel_est = a_rel_est.fillna(0.0)

    # Compute Projected Min Distance
    min_dist_proj = compute_quadratic_min_dist(
        df_merged["distance"].values,
        v_rel_est.values,
        a_rel_est.values,
        lookahead=Config.GATING["LOOKAHEAD_STEPS"] * Config.GATING["TIME_STEP"],
    )
    df_merged["gating_min_dist"] = min_dist_proj

    # 6. Spectral-Kinematic Window Features
    logger.info("Generating Spectral-Kinematic features...")

    # Construct window matrix for rel_accel using shifts
    window_size = Config.FEATURES["WINDOW_SIZE"]
    step_col = df_merged["step"]
    rel_accel_col = df_merged["rel_accel"]

    shock_inputs = []

    # Gather columns for t-10 to t+10
    for k in range(-window_size, window_size + 1):
        # shift(-k) looks forward if k is positive, backward if k is negative
        # We want index i to contain value at i+k
        shifted_acc = rel_accel_col.shift(-k)
        shifted_step = step_col.shift(-k)
        shifted_group = group_id.shift(-k)

        # Validate shift (must be same group and contiguous time)
        valid_shift = (shifted_group == group_id) & ((shifted_step - step_col) == k)

        # Fill invalid shifts with 0
        shifted_acc[~valid_shift] = 0.0
        shock_inputs.append(shifted_acc.values)

    shock_matrix = np.stack(shock_inputs, axis=1)  # (N, 21)

    # Apply High Pass Filter (Remove Trend)
    # Manual convolution for Trend (Smoothing)
    # Pad axis 1 to handle edges for window=3
    padded = np.pad(shock_matrix, ((0, 0), (1, 1)), mode="edge")
    # Rolling sum of 3 divided by 3
    trend_matrix = (padded[:, 0:21] + padded[:, 1:22] + padded[:, 2:23]) / 3.0

    high_freq_matrix = shock_matrix - trend_matrix

    # Compute Energy
    df_merged["spectral_energy"] = compute_spectral_energy(high_freq_matrix)

    # 7. Final Gating & Selection
    if Config.GATING["ENABLED"] and dataset_type != "test":
        # Keep Ground OR (Player-Player AND Reachable)
        keep_mask = (is_ground) | (
            df_merged["gating_min_dist"] < Config.GATING["DIST_THRESHOLD"]
        )
        logger.info(
            f"Gating: Dropping {len(df_merged) - keep_mask.sum()} rows out of {len(df_merged)}."
        )
        df_merged = df_merged[keep_mask].copy()

    feature_cols = [
        "distance",
        "rel_speed",
        "rel_accel",
        "ke_p1",
        "ke_p2",
        "spectral_energy",
        "gating_min_dist",
    ]

    if "contact" in df_merged.columns:
        feature_cols.append("contact")

    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ]
    final_cols = meta_cols + feature_cols

    df_final = df_merged[final_cols]

    # Save
    save_cache_parquet(df_final, cache_file)
    logger.info(f"Saved {len(df_final)} rows to {cache_file}")

    # Cleanup
    del df_track, df_track_p1, df_track_p2, df_merged, shock_matrix
    gc.collect()

    return df_final
