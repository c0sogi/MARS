import os
import pandas as pd
import numpy as np
import gc
from library.config import Config
from library.utils import setup_logger, get_cache_path, ensure_dir
from library.data_loader import load_metadata, load_tracking

# Initialize Logger
logger = setup_logger("feature_engineering")


def calculate_quadratic_min_dist(df, window_seconds=1.0):
    """
    Estimates the minimum distance between players within a time window
    using a quadratic motion model based on t=0 state.
    r(t) = r0 + v*t + 0.5*a*t^2
    """
    # Relative vectors at t=0
    rx = df["x_position_p1"] - df["x_position_p2"]
    ry = df["y_position_p1"] - df["y_position_p2"]
    vx = df["speed_x_p1"] - df["speed_x_p2"]
    vy = df["speed_y_p1"] - df["speed_y_p2"]
    ax = df["acc_x_p1"] - df["acc_x_p2"]
    ay = df["acc_y_p1"] - df["acc_y_p2"]

    # We evaluate distance at discrete points in the window [-1, 1]
    # Solving for exact min of quartic polynomial is expensive, sampling is sufficient for gating.
    t_points = np.linspace(-window_seconds, window_seconds, 5)

    min_sq_dist = np.full(len(df), np.inf)

    for t in t_points:
        # Projected relative position at time t
        # p(t) = p0 + v0*t + 0.5*a0*t^2
        pred_rx = rx + vx * t + 0.5 * ax * (t**2)
        pred_ry = ry + vy * t + 0.5 * ay * (t**2)
        dist_sq = pred_rx**2 + pred_ry**2
        min_sq_dist = np.minimum(min_sq_dist, dist_sq)

    return np.sqrt(min_sq_dist)


def decompose_vector(v_x, v_y, basis_x, basis_y):
    """
    Projects vector V onto Basis U (Radial) and Orthogonal Basis U_perp (Tangential).
    """
    # Radial component: Dot product
    radial = v_x * basis_x + v_y * basis_y

    # Tangential component: Cross product analog (2D)
    # U_perp = (-basis_y, basis_x)
    tangential = v_x * (-basis_y) + v_y * basis_x

    return radial, tangential


def generate_features(
    split: str, load_cached_data: bool = True, debug_sample: int = None
):
    """
    Main function to generate features for a given split.
    Implements Relaxed Quadratic Gating and Dynamic-Basis Decoupled Feature Engineering.
    """
    # 1. Check Cache
    cache_path = (
        Config.CACHE_TRAIN_FEATURES
        if split == "train"
        else Config.CACHE_VAL_FEATURES if split == "val" else Config.CACHE_TEST_FEATURES
    )

    if debug_sample is not None:
        cache_path = cache_path.replace(".parquet", f"_sample_{debug_sample}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    logger.info(f"Generating features for {split} (Debug: {debug_sample})...")

    # 2. Load Raw Data
    df_meta = load_metadata(split)
    df_track = load_tracking(split)

    if debug_sample:
        logger.info(f"Sampling {debug_sample} rows for debugging...")
        df_meta = df_meta.iloc[:debug_sample].copy()

    # 3. Preprocess Tracking Data
    # Convert polar velocity/accel to Cartesian for vector math
    # orientation/direction are in degrees. 0 is Y axis (usually), increasing clockwise?
    # NFL Data: 0 is Y-axis (short axis), 90 is X-axis.
    # Usually: x component = sin(theta), y component = cos(theta) if 0 is North (Y).
    # Standard math: 0 is X, 90 is Y.
    # We will assume standard NFL tracking convention: 0 is along Y (short axis), 90 is along X (long axis).
    # rad = (90 - deg) * pi / 180 converts to standard math angle.
    # However, simpler: dir=0 -> y+, dir=90 -> x+.
    # vx = speed * sin(dir), vy = speed * cos(dir)

    logger.info("Preprocessing tracking vectors...")
    rad = np.radians(df_track["direction"])
    df_track["speed_x"] = df_track["speed"] * np.sin(rad)
    df_track["speed_y"] = df_track["speed"] * np.cos(rad)

    # Acceleration is magnitude. We don't have accel direction explicitly in provided columns?
    # Provided cols: speed, distance, orientation, direction, acceleration, sa.
    # We assume acceleration acts in the direction of motion (or use 'sa' signed accel).
    # Let's approximate accel vector aligns with speed vector direction for gating.
    df_track["acc_x"] = df_track["acceleration"] * np.sin(rad)
    df_track["acc_y"] = df_track["acceleration"] * np.cos(rad)

    # Index for fast merging
    # We need to merge on (game_play, step, nfl_player_id)
    # To optimize, we'll keep df_track as is and use merge.

    # 4. Gating (Stage 0)
    # We need current step info to gate.
    logger.info("Performing Relaxed Quadratic Gating...")

    # Prepare P2 ID for merging (handle 'G')
    df_meta["nfl_player_id_2_int"] = pd.to_numeric(
        df_meta["nfl_player_id_2"], errors="coerce"
    )
    is_ground = df_meta["nfl_player_id_2"] == "G"

    # Merge T=0 for P1
    track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed_x",
        "speed_y",
        "acc_x",
        "acc_y",
    ]

    df_gate = df_meta[
        ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2_int"]
    ].copy()

    df_gate = df_gate.merge(
        df_track[track_cols],
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    ).rename(
        columns={c: f"{c}_p1" for c in track_cols if c not in ["game_play", "step"]}
    )

    # Merge T=0 for P2 (only non-ground)
    df_gate = df_gate.merge(
        df_track[track_cols],
        left_on=["game_play", "step", "nfl_player_id_2_int"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_p2"),
    ).rename(
        columns={c: f"{c}_p2" for c in track_cols if c not in ["game_play", "step"]}
    )

    # Calculate Min Distance for P-P
    # Fill NaNs for P2 (Ground) with 0 to avoid errors, though we won't use them for filtering
    df_gate = df_gate.fillna(0)

    predicted_min_dist = calculate_quadratic_min_dist(df_gate, window_seconds=1.0)

    # Filter: Keep if Ground OR (MinDist < Threshold)
    keep_mask = is_ground | (predicted_min_dist < Config.GATING_THRESHOLD)

    logger.info(
        f"Gating complete. Retained {keep_mask.sum()} / {len(df_meta)} rows ({(keep_mask.sum()/len(df_meta)):.2%})."
    )
    df_meta = df_meta[keep_mask].reset_index(drop=True)

    # Clean up memory
    del df_gate, predicted_min_dist, keep_mask
    gc.collect()

    # 5. Dynamic-Basis Feature Engineering (Stage 1)
    # We iterate through the window
    feature_dfs = []

    # Pre-calculate P2 integer ID again for the filtered df
    df_meta["nfl_player_id_2_int"] = pd.to_numeric(
        df_meta["nfl_player_id_2"], errors="coerce"
    )

    # Columns to extract for features
    # We need pos, vel, acc for projection
    step_track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed_x",
        "speed_y",
        "acc_x",
        "acc_y",
        "speed",
        "acceleration",
        "orientation",
        "direction",
    ]

    for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        # Current relative step
        current_step_col = df_meta["step"] + offset

        # Create a temporary merge key
        temp_meta = df_meta[
            ["game_play", "nfl_player_id_1", "nfl_player_id_2_int", "nfl_player_id_2"]
        ].copy()
        temp_meta["step_lookup"] = current_step_col

        # Merge P1
        merged = temp_meta.merge(
            df_track[step_track_cols],
            left_on=["game_play", "step_lookup", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["step", "nfl_player_id"])

        # Rename P1
        p1_cols = {
            c: f"{c}_p1"
            for c in step_track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        merged = merged.rename(columns=p1_cols)

        # Merge P2
        merged = merged.merge(
            df_track[step_track_cols],
            left_on=["game_play", "step_lookup", "nfl_player_id_2_int"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["step", "nfl_player_id"])

        # Rename P2
        p2_cols = {
            c: f"{c}_p2"
            for c in step_track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        merged = merged.rename(columns=p2_cols)

        # --- Feature Calculation ---

        # 1. Determine Basis Vector (u_x, u_y)
        # Initialize with zeros
        u_x = np.zeros(len(merged))
        u_y = np.zeros(len(merged))

        # Mask for Ground vs Player
        mask_g = merged["nfl_player_id_2"] == "G"
        mask_p = ~mask_g

        # P-P Basis: Unit vector from P2 to P1
        # Vector = P1 - P2
        dx = merged.loc[mask_p, "x_position_p1"] - merged.loc[mask_p, "x_position_p2"]
        dy = merged.loc[mask_p, "y_position_p1"] - merged.loc[mask_p, "y_position_p2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Handle zero distance (overlap) - fallback to P1 velocity or arbitrary (1,0)
        # Add epsilon to avoid div by zero
        dist = dist.replace(0, 1e-6)

        u_x[mask_p] = dx / dist
        u_y[mask_p] = dy / dist

        # P-G Basis: Unit vector of P1 Velocity
        # Vector = V1
        v1_x = merged.loc[mask_g, "speed_x_p1"].fillna(0)
        v1_y = merged.loc[mask_g, "speed_y_p1"].fillna(0)
        v1_mag = np.sqrt(v1_x**2 + v1_y**2).replace(0, 1e-6)

        u_x[mask_g] = v1_x / v1_mag
        u_y[mask_g] = v1_y / v1_mag

        # 2. Project Vectors
        # P1 Velocity
        v1_rad, v1_tan = decompose_vector(
            merged["speed_x_p1"], merged["speed_y_p1"], u_x, u_y
        )
        # P1 Accel
        a1_rad, a1_tan = decompose_vector(
            merged["acc_x_p1"], merged["acc_y_p1"], u_x, u_y
        )

        # P2 Velocity (0 for Ground)
        v2_rad, v2_tan = decompose_vector(
            merged["speed_x_p2"].fillna(0), merged["speed_y_p2"].fillna(0), u_x, u_y
        )
        # P2 Accel
        a2_rad, a2_tan = decompose_vector(
            merged["acc_x_p2"].fillna(0), merged["acc_y_p2"].fillna(0), u_x, u_y
        )

        # 3. Distance Feature
        # Real distance for P-P, Sentinel for P-G
        final_dist = np.full(len(merged), Config.GROUND_DISTANCE_SENTINEL)
        final_dist[mask_p] = dist

        # 4. Store Features for this Lag
        lag_df = pd.DataFrame(
            {
                f"dist_lag{offset}": final_dist,
                f"p1_v_rad_lag{offset}": v1_rad,
                f"p1_v_tan_lag{offset}": v1_tan,
                f"p1_a_rad_lag{offset}": a1_rad,
                f"p1_a_tan_lag{offset}": a1_tan,
                f"p2_v_rad_lag{offset}": v2_rad,
                f"p2_v_tan_lag{offset}": v2_tan,
                f"p2_a_rad_lag{offset}": a2_rad,
                f"p2_a_tan_lag{offset}": a2_tan,
                # Raw scalar features
                f"p1_speed_lag{offset}": merged["speed_p1"],
                f"p2_speed_lag{offset}": merged["speed_p2"].fillna(0),
                f"p1_orient_lag{offset}": merged[
                    "orientation_p1"
                ],  # Keeping orientation might help tree models implicitly
                f"p2_orient_lag{offset}": merged["orientation_p2"].fillna(0),
            }
        )

        # Handle NaNs (e.g. if lag is outside play duration)
        # Fill with 0 or forward fill? 0 is safer for tree models with centered features.
        lag_df = lag_df.fillna(0)

        feature_dfs.append(lag_df)

    # Concatenate all lag features horizontally
    logger.info("Concatenating temporal features...")
    X = pd.concat(feature_dfs, axis=1)

    # Add Metadata columns required for training/submission
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ]
    if "contact" in df_meta.columns:
        meta_cols.append("contact")

    final_df = pd.concat([df_meta[meta_cols].reset_index(drop=True), X], axis=1)

    # 6. Save to Cache
    ensure_dir(cache_path)
    logger.info(f"Saving features to {cache_path}...")
    final_df.to_parquet(cache_path, index=False)

    return final_df
