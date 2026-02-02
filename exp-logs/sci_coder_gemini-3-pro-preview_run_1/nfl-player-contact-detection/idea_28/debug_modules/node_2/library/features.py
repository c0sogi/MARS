import pandas as pd
import numpy as np
import os
import logging
from library.config import Config
from library.utils import (
    get_cache_path,
    save_to_parquet,
    load_from_parquet,
    check_cache_exists,
    generate_config_hash,
)

# Setup logger
logger = logging.getLogger("NFL_Contact_Detection")


def compute_spectral_shock(tracking_df):
    """
    Computes the Spectral Shock (RMS of High-Passed Acceleration) for each player.
    Operates on the full tracking dataframe before merging.
    """
    # Ensure sorted order for rolling operations
    tracking_df = tracking_df.sort_values(by=["game_play", "nfl_player_id", "step"])

    # We use 'acceleration' magnitude as the base signal.
    target_col = "acceleration"

    window = Config.SPECTRAL_WINDOW_SIZE

    # Function to apply to each group
    # High-pass: Signal - Low_Freq(Signal)
    # Spectral Energy: RMS(High_Pass)
    def get_spectral(x):
        # Calculate low frequency component (trend)
        low_freq = x.rolling(window=window, center=True, min_periods=1).mean()
        # High pass component (noise/shock)
        high_freq = x - low_freq
        # RMS energy of the high frequency component
        energy = np.sqrt(
            (high_freq**2).rolling(window=window, center=True, min_periods=1).mean()
        )
        return energy.fillna(0.0)

    logger.info("Computing Spectral Shock on tracking data...")

    # Fill NaNs in target column to prevent propagation
    if target_col in tracking_df.columns:
        tracking_df[target_col] = tracking_df[target_col].fillna(0.0)
    else:
        # Fallback if acceleration is missing (unlikely given description)
        tracking_df[target_col] = 0.0

    # Apply transform grouped by player within play
    spectral_vals = tracking_df.groupby(["game_play", "nfl_player_id"])[
        target_col
    ].transform(get_spectral)

    tracking_df["spectral_energy"] = spectral_vals

    return tracking_df


def apply_quadratic_gating(df):
    """
    Filters player pairs based on a quadratic trajectory approximation.
    Logic: d(t) = d0 + v_rel*t + 0.5*a_rel*t^2
    Retains pairs where min(d(t)) < Threshold in the window.
    Always retains Ground interactions.
    """
    logger.info("Applying Relaxed Quadratic Gating...")

    # Identify Ground interactions (Always keep)
    is_ground = df["nfl_player_id_2"] == "G"

    # If all are ground, return immediately
    if is_ground.all():
        return df

    # Constants
    dt = 0.1  # seconds per step
    window_steps = Config.GATING_WINDOW_STEPS
    # Time offsets from current step: e.g., [-0.5, -0.4, ..., 0.5]
    time_offsets = np.arange(-window_steps, window_steps + 1) * dt

    # Helper to extract kinematic components
    # Assuming standard mathematical convention for projection (0=East, 90=North)
    # or consistent relative usage.
    # Note: NFL tracking 'direction' is 0=Y(North), 90=X(East) usually,
    # but as long as we use sin/cos consistently for both players, relative vector magnitude is correct.
    def get_vecs(suffix):
        s = df[f"speed{suffix}"].fillna(0)
        d_rad = np.radians(df[f"direction{suffix}"].fillna(0))
        a = df[f"acceleration{suffix}"].fillna(0)

        # Project speed and accel into X/Y components
        # We assume acceleration vector aligns with motion direction for this approximation
        vx = s * np.sin(d_rad)
        vy = s * np.cos(d_rad)
        ax = a * np.sin(d_rad)
        ay = a * np.cos(d_rad)

        px = df[f"x_position{suffix}"].fillna(0)
        py = df[f"y_position{suffix}"].fillna(0)

        return px, py, vx, vy, ax, ay

    px1, py1, vx1, vy1, ax1, ay1 = get_vecs("_p1")
    px2, py2, vx2, vy2, ax2, ay2 = get_vecs("_p2")

    # Relative Kinematics
    rx = px1 - px2
    ry = py1 - py2
    rvx = vx1 - vx2
    rvy = vy1 - vy2
    rax = ax1 - ax2
    ray = ay1 - ay2

    # Compute minimum distance over the time window
    # Initialize with infinity
    min_dists = np.full(len(df), np.inf)

    # Vectorized loop over time offsets
    for t in time_offsets:
        # Predicted relative position at time t
        pred_rx = rx + rvx * t + 0.5 * rax * (t**2)
        pred_ry = ry + rvy * t + 0.5 * ray * (t**2)
        dist_t = np.sqrt(pred_rx**2 + pred_ry**2)
        min_dists = np.minimum(min_dists, dist_t)

    # Create Filter Mask
    # Keep if Ground Interaction OR Projected Distance < Threshold
    mask = is_ground | (min_dists < Config.GATING_DIST_THRESH)

    dropped_count = len(df) - mask.sum()
    logger.info(
        f"Gating dropped {dropped_count} rows ({dropped_count/len(df)*100:.2f}%)."
    )

    return df[mask].copy()


def project_vectors(df):
    """
    Decomposes velocity and acceleration into Dual-Basis components.
    Case A (Player-Player): Basis = Collision Axis.
    Case B (Player-Ground): Basis = Motion Axis.
    """
    logger.info("Projecting vectors into Dual-Basis...")

    is_ground = df["nfl_player_id_2"] == "G"

    # Initialize basis vectors u = (ux, uy)
    ux = np.zeros(len(df), dtype=np.float32)
    uy = np.zeros(len(df), dtype=np.float32)

    # --- 1. Define Basis for Player-Player (Collision Axis) ---
    # u = (p1 - p2) / |p1 - p2|
    dx = df["x_position_p1"] - df["x_position_p2"]
    dy = df["y_position_p1"] - df["y_position_p2"]
    dist = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    mask_pp = (~is_ground) & (dist > 1e-6)
    if mask_pp.any():
        ux[mask_pp] = dx[mask_pp] / dist[mask_pp]
        uy[mask_pp] = dy[mask_pp] / dist[mask_pp]

    # --- 2. Define Basis for Player-Ground (Motion Axis) ---
    # u = v1 / |v1|
    s1 = df["speed_p1"]
    d1_rad = np.radians(df["direction_p1"])

    # Calculate velocity components (using sin/cos for NFL coords: 0=Y, 90=X)
    vx1_raw = s1 * np.sin(d1_rad)
    vy1_raw = s1 * np.cos(d1_rad)

    mask_pg = is_ground & (s1 > 1e-6)
    if mask_pg.any():
        ux[mask_pg] = vx1_raw[mask_pg] / s1[mask_pg]
        uy[mask_pg] = vy1_raw[mask_pg] / s1[mask_pg]

    # Fallback for stationary Player-Ground: Use Orientation (Facing)
    mask_pg_stat = is_ground & (s1 <= 1e-6)
    if mask_pg_stat.any():
        o1_rad = np.radians(df["orientation_p1"][mask_pg_stat])
        ux[mask_pg_stat] = np.sin(o1_rad)
        uy[mask_pg_stat] = np.cos(o1_rad)

    # --- 3. Project P1 Vectors ---
    # Construct raw acceleration vector (assuming alignment with direction)
    a1 = df["acceleration_p1"]
    ax1_raw = a1 * np.sin(d1_rad)
    ay1_raw = a1 * np.cos(d1_rad)

    # Comp 1 (Parallel): v . u
    # Comp 2 (Perpendicular): v . u_perp, where u_perp = (-uy, ux)
    df["v_comp1_p1"] = vx1_raw * ux + vy1_raw * uy
    df["v_comp2_p1"] = vx1_raw * (-uy) + vy1_raw * ux

    df["a_comp1_p1"] = ax1_raw * ux + ay1_raw * uy
    df["a_comp2_p1"] = ax1_raw * (-uy) + ay1_raw * ux

    # --- 4. Project P2 Vectors (Only relevant for P-P) ---
    # For P-G, P2 is Ground (dummy values 0), so projections will be 0.
    s2 = df["speed_p2"].fillna(0)
    d2_rad = np.radians(df["direction_p2"].fillna(0))
    a2 = df["acceleration_p2"].fillna(0)

    vx2_raw = s2 * np.sin(d2_rad)
    vy2_raw = s2 * np.cos(d2_rad)
    ax2_raw = a2 * np.sin(d2_rad)
    ay2_raw = a2 * np.cos(d2_rad)

    df["v_comp1_p2"] = vx2_raw * ux + vy2_raw * uy
    df["v_comp2_p2"] = vx2_raw * (-uy) + vy2_raw * ux

    df["a_comp1_p2"] = ax2_raw * ux + ay2_raw * uy
    df["a_comp2_p2"] = ax2_raw * (-uy) + ay2_raw * ux

    return df


def generate_features(split="train", load_cached_data=True, debug=False):
    """
    Main pipeline to generate Dual-Basis features.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache.
        debug (bool): If True, process a subset for debugging.

    Returns:
        pd.DataFrame: Processed feature dataframe.
    """
    # 1. Config & Cache Check
    config_dict = {
        "split": split,
        "gating_window": Config.GATING_WINDOW_STEPS,
        "gating_thresh": Config.GATING_DIST_THRESH,
        "spectral_window": Config.SPECTRAL_WINDOW_SIZE,
        "sentinel_dist": Config.SENTINEL_DISTANCE_VALUE,
        "debug": debug,
    }
    config_hash = generate_config_hash(config_dict)
    cache_file = f"features_{split}"

    if load_cached_data and check_cache_exists(
        get_cache_path(cache_file, config_hash, ".parquet")
    ):
        logger.info(f"Loading cached features for {split}...")
        return load_from_parquet(get_cache_path(cache_file, config_hash, ".parquet"))

    logger.info(f"Generating features for {split} from scratch...")

    # 2. Load Metadata
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    df_meta = pd.read_csv(meta_path)
    if debug:
        df_meta = df_meta.sample(n=10000, random_state=Config.SEED)

    # 3. Load & Process Tracking Data
    track_path = (
        Config.TEST_TRACKING_PATH if split == "test" else Config.TRAIN_TRACKING_PATH
    )
    df_tracking = pd.read_csv(track_path)

    # Compute Spectral Shock (Intrinsic Player Feature)
    df_tracking = compute_spectral_shock(df_tracking)

    # 4. Merge Data
    logger.info("Merging metadata and tracking data...")

    # Ensure join keys match types
    df_meta["game_play"] = df_meta["game_play"].astype(str)
    df_meta["step"] = df_meta["step"].astype(int)
    df_tracking["game_play"] = df_tracking["game_play"].astype(str)
    df_tracking["step"] = df_tracking["step"].astype(int)

    # Select columns to merge
    track_cols = Config.RAW_TRACKING_COLS + ["spectral_energy"]
    # Filter to ensure columns exist in source
    track_cols = [c for c in track_cols if c in df_tracking.columns]

    # --- Merge Player 1 ---
    df_merged = pd.merge(
        df_meta,
        df_tracking[track_cols],
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns
    rename_p1 = {c: f"{c}_p1" for c in track_cols if c not in ["game_play", "step"]}
    df_merged = df_merged.rename(columns=rename_p1)

    # --- Merge Player 2 (Handle Ground) ---
    mask_ground = df_merged["nfl_player_id_2"] == "G"

    # Split dataset
    df_pg = df_merged[mask_ground].copy()
    df_pp = df_merged[~mask_ground].copy()

    # Process Player-Player
    if not df_pp.empty:
        df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)
        df_pp = pd.merge(
            df_pp,
            df_tracking[track_cols],
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        rename_p2 = {c: f"{c}_p2" for c in track_cols if c not in ["game_play", "step"]}
        df_pp = df_pp.rename(columns=rename_p2)

    # Process Player-Ground (Fill P2 with 0)
    for c in track_cols:
        if c not in ["game_play", "step"]:
            df_pg[f"{c}_p2"] = 0.0

    # Recombine
    df_full = pd.concat([df_pp, df_pg], axis=0)

    # Fill NaNs in tracking data (Crucial for Test set where we cannot drop rows)
    # P1 data might be NaN if tracking is missing for that step
    feat_cols = [c for c in df_full.columns if "_p1" in c or "_p2" in c]
    df_full[feat_cols] = df_full[feat_cols].fillna(0.0)

    # 5. Gating (Stage 0)
    # We only drop rows for Train/Val to improve balance.
    # For Test, we must predict for every row, so we skip dropping.
    if split != "test":
        df_full = apply_quadratic_gating(df_full)

    # 6. Feature Engineering (Stage 1)
    # Calculate Euclidean Distance
    dx = df_full["x_position_p1"] - df_full["x_position_p2"]
    dy = df_full["y_position_p1"] - df_full["y_position_p2"]
    df_full["distance"] = np.sqrt(dx**2 + dy**2)

    # Apply Sentinel Value for Ground Distance
    df_full.loc[df_full["nfl_player_id_2"] == "G", "distance"] = (
        Config.SENTINEL_DISTANCE_VALUE
    )

    # Dual Basis Projections
    df_full = project_vectors(df_full)

    # 7. Final Column Selection
    # Keep identifiers, features, and target (if available)
    keep_cols = ["contact_id", "game_play", "step"] + Config.MODEL_FEATURES
    if "contact" in df_full.columns:
        keep_cols.append("contact")

    df_final = df_full[keep_cols].copy()

    # 8. Save to Cache
    save_to_parquet(df_final, get_cache_path(cache_file, config_hash, ".parquet"))

    return df_final
