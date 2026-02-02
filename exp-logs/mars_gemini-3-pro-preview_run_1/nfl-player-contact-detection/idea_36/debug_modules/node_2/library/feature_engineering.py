import os
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from library import config, utils, data_processing

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
# Define the lags based on window size
LAGS = list(range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1))
TOTAL_LAGS = len(LAGS)

# Columns required from tracking data
TRACKING_COLS = [
    "game_play",
    "step",
    "nfl_player_id",
    "x_position",
    "y_position",
    "speed",
    "direction",
    "acceleration",
    "sa",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _load_full_tracking(path):
    """
    Loads the full tracking dataset for windowed feature extraction.
    Optimizes memory usage.
    """
    print(f"Loading full tracking data from {path}...")
    df = pd.read_csv(path, usecols=TRACKING_COLS)
    df = utils.reduce_mem_usage(df)

    # Pre-calculate velocity components to save time during merges
    # v_x = speed * sin(direction)
    # v_y = speed * cos(direction)
    # Note: direction is 0 at North (Y-axis), increasing clockwise.
    # Standard math: 0 at East (X-axis), increasing counter-clockwise.
    # NFL tracking: 0=Y, 90=X.
    # v_x = speed * sin(theta)
    # v_y = speed * cos(theta)

    # Fill NaNs
    df["speed"] = df["speed"].fillna(0)
    # Convert direction to radians (0 is North/Y, increasing clockwise)
    if "direction" in df.columns:
        df["direction_rad"] = np.deg2rad(df["direction"].fillna(0))
    else:
        df["direction_rad"] = 0.0
    df["acceleration"] = df["acceleration"].fillna(0)

    # Compute components
    df["v_x"] = df["speed"] * np.sin(df["direction_rad"])
    df["v_y"] = df["speed"] * np.cos(df["direction_rad"])

    # Drop raw direction/speed to save memory if not needed,
    # but we might need them for raw features. Keeping them for now.

    return df


def _expand_with_lags(df_meta):
    """
    Expands the metadata DataFrame to include all timesteps in the window.
    Returns a DataFrame with N * 21 rows.
    """
    # Create a list of DataFrames, one for each lag
    dfs = []
    for lag in LAGS:
        df_lag = df_meta[
            ["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
        ].copy()
        df_lag["lag"] = lag
        df_lag["step_window"] = df_lag["step"] + lag
        dfs.append(df_lag)

    return pd.concat(dfs, axis=0, ignore_index=True)


def _merge_window_tracking(df_expanded, df_tracking):
    """
    Merges tracking data for P1 and P2 onto the expanded window DataFrame.
    """
    # ---------------------------------------------------------
    # Merge Player 1
    # ---------------------------------------------------------
    df_merged = pd.merge(
        df_expanded,
        df_tracking,
        left_on=["game_play", "step_window", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_p1"),
    )
    # Rename columns to have _p1 suffix explicitly
    rename_cols = {
        "x_position": "x_p1",
        "y_position": "y_p1",
        "v_x": "vx_p1",
        "v_y": "vy_p1",
        "acceleration": "a_p1",
        "sa": "sa_p1",
        "direction_rad": "dir_p1",
    }
    df_merged = df_merged.rename(columns=rename_cols)

    # Cleanup redundant columns
    drop_cols = ["step_y", "nfl_player_id"]  # step_x is step_window or original step?
    # Pandas merge: left keys are preserved. right keys might be duplicated.
    # We joined on step_window (left) and step (right).
    # The result usually keeps the left key.
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # ---------------------------------------------------------
    # Merge Player 2
    # ---------------------------------------------------------
    # Handle 'G' (Ground) separately implicitly:
    # If nfl_player_id_2 is 'G', the merge will fail (produce NaNs).
    # We must ensure nfl_player_id_2 is same type as tracking nfl_player_id (float/int).
    # 'G' is string. Tracking ID is int.
    # We create a temporary column for merging that is numeric or NaN.

    df_merged["p2_merge_key"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    df_merged = pd.merge(
        df_merged,
        df_tracking,
        left_on=["game_play", "step_window", "p2_merge_key"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_p2"),
    )

    rename_cols_p2 = {
        "x_position": "x_p2",
        "y_position": "y_p2",
        "v_x": "vx_p2",
        "v_y": "vy_p2",
        "acceleration": "a_p2",
        "sa": "sa_p2",
        "direction_rad": "dir_p2",
    }
    df_merged = df_merged.rename(columns=rename_cols_p2)

    # Drop redundant
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])
    df_merged = df_merged.drop(columns=["p2_merge_key"])

    # ---------------------------------------------------------
    # Impute Missing / Ground
    # ---------------------------------------------------------
    # For Ground (p2='G') or missing tracking, fill P2 with 0s.
    # Note: P2 Position for Ground is undefined, but we handle it in basis calc.
    p2_cols = ["x_p2", "y_p2", "vx_p2", "vy_p2", "a_p2", "sa_p2", "dir_p2"]
    for col in p2_cols:
        df_merged[col] = df_merged[col].fillna(0.0)

    # For P1 missing tracking, fill with 0s (edge case)
    p1_cols = ["x_p1", "y_p1", "vx_p1", "vy_p1", "a_p1", "sa_p1", "dir_p1"]
    for col in p1_cols:
        df_merged[col] = df_merged[col].fillna(0.0)

    return df_merged


def _compute_dynamic_basis_features(df):
    """
    Computes the Dynamic-Basis Relative Kinematics for each row (timestep).
    """
    # 1. Determine Basis Vectors
    # P-P Basis: Unit vector from P2 to P1
    dx = df["x_p1"] - df["x_p2"]
    dy = df["y_p1"] - df["y_p2"]
    dist = np.sqrt(dx**2 + dy**2)

    # Avoid div by zero
    dist_safe = np.where(dist < 1e-6, 1e-6, dist)

    # Basis u (Radial)
    ux = dx / dist_safe
    uy = dy / dist_safe

    # P-G Basis: Unit vector of P1 velocity
    # If P2 is Ground ('G'), override basis
    is_ground = df["nfl_player_id_2"] == "G"

    v1_mag = np.sqrt(df["vx_p1"] ** 2 + df["vy_p1"] ** 2)
    v1_mag_safe = np.where(v1_mag < 1e-6, 1e-6, v1_mag)

    # If ground, basis is P1 velocity direction
    ux = np.where(is_ground, df["vx_p1"] / v1_mag_safe, ux)
    uy = np.where(is_ground, df["vy_p1"] / v1_mag_safe, uy)

    # If ground and P1 has 0 velocity, default to (1, 0) to avoid NaNs
    zero_vel_ground = is_ground & (v1_mag < 1e-6)
    ux = np.where(zero_vel_ground, 1.0, ux)
    uy = np.where(zero_vel_ground, 0.0, uy)

    # Orthogonal Basis u_perp (Tangential)
    # Rotate 90 degrees: (x, y) -> (-y, x)
    ux_perp = -uy
    uy_perp = ux

    # 2. Relative Kinematics
    # v_rel = v1 - v2
    dvx = df["vx_p1"] - df["vx_p2"]
    dvy = df["vy_p1"] - df["vy_p2"]

    # a_rel = a1 - a2 (Approximate acceleration vector using 'acceleration' magnitude and direction?
    # The tracking data gives scalar 'acceleration' and 'direction'.
    # Assuming acceleration is in direction of motion is a simplification, but standard for this data.
    # Better: Use finite difference of velocity?
    # Given 'acceleration' column exists, let's use it projected on velocity direction.
    ax1 = df["a_p1"] * np.sin(df["dir_p1"])
    ay1 = df["a_p1"] * np.cos(df["dir_p1"])
    ax2 = df["a_p2"] * np.sin(df["dir_p2"])
    ay2 = df["a_p2"] * np.cos(df["dir_p2"])

    dax = ax1 - ax2
    day = ay1 - ay2

    # 3. Projections
    # Radial Velocity: v_rel dot u
    df["v_rad"] = dvx * ux + dvy * uy
    # Tangential Velocity: v_rel dot u_perp
    df["v_tan"] = dvx * ux_perp + dvy * uy_perp

    # Radial Acceleration
    df["a_rad"] = dax * ux + day * uy
    # Tangential Acceleration
    df["a_tan"] = dax * ux_perp + day * uy_perp

    # Distance (Sentinel for Ground)
    df["dist"] = np.where(is_ground, -1.0, dist)

    # Relative Orientation (difference in facing angle)
    # We use cos/sin of difference to avoid wrap-around issues, or just raw diff
    # Let's use cos(theta1 - theta2) as a similarity metric
    # For Ground, orientation of P2 is 0 (undefined).
    df["orientation_rel"] = np.cos(df["dir_p1"] - df["dir_p2"])
    df["orientation_rel"] = np.where(is_ground, 0.0, df["orientation_rel"])

    return df


def _calculate_scalars(df_wide):
    """
    Calculates physics primitives (scalars) from the wide dataframe.
    """
    # Min Distance over window
    # Columns are dist_t-10 ... dist_t+10
    dist_cols = [f"dist_t{lag:+d}" for lag in LAGS]

    # Filter out sentinel values (-1.0) for min calculation if mixed (unlikely in one row)
    # But for Ground rows, all dists are -1.0. Min is -1.0. Correct.
    # For P-P, dists are positive.

    # We can just take min across columns
    df_wide["min_dist"] = df_wide[dist_cols].min(axis=1)

    # Time to Collision (TTC) at t=0
    # TTC = dist / -v_rad (if v_rad < 0, i.e., closing)
    dist_0 = df_wide["dist_t+0"]
    v_rad_0 = df_wide["v_rad_t+0"]

    ttc = dist_0 / (-v_rad_0 + 1e-6)
    # Filter: only valid if closing (v_rad < 0) and not ground
    is_closing = v_rad_0 < -0.1
    is_ground = dist_0 == -1.0

    df_wide["time_to_collision"] = np.where(
        is_closing & (~is_ground), ttc, 10.0
    )  # Default large value

    # Jerk Magnitude (Derivative of Acceleration)
    # Approx: |a_rad_t+1 - a_rad_t-1| / 0.2s
    # Let's take average jerk over window or max jerk?
    # Strategy says "Jerk". Let's use max magnitude of jerk in window.
    # We'll compute jerk for the central frame t=0: (a_t+1 - a_t-1) / 0.2
    if 1 in LAGS and -1 in LAGS:
        ax0 = df_wide["a_rad_t+0"]  # Using radial acceleration as proxy
        # Better: Magnitude of vector jerk.
        # Let's keep it simple: change in total acceleration magnitude
        # a_mag = sqrt(a_rad^2 + a_tan^2).
        # We don't have a_mag columns readily pivoted.
        # Let's use a_rad derivative at t=0.
        jerk = (df_wide["a_rad_t+1"] - df_wide["a_rad_t-1"]) / 0.2
        df_wide["jerk_mag"] = np.abs(jerk)
    else:
        df_wide["jerk_mag"] = 0.0

    # Angular Jerk
    # Change in v_tan or orientation?
    # Let's use derivative of v_tan (shear jerk)
    if 1 in LAGS and -1 in LAGS:
        ang_jerk = (df_wide["v_tan_t+1"] - df_wide["v_tan_t-1"]) / 0.2
        df_wide["angular_jerk"] = np.abs(ang_jerk)
    else:
        df_wide["angular_jerk"] = 0.0

    return df_wide


def _pivot_to_wide(df_long):
    """
    Pivots the long-format DataFrame (with lags) to a wide format (one row per contact_id).
    """
    # We want to pivot on 'contact_id'
    # Values to pivot: dist, v_rad, v_tan, a_rad, a_tan, orientation_rel
    values = config.DYNAMIC_FEATURE_BASES

    # Pivot
    df_wide = df_long.pivot_table(index="contact_id", columns="lag", values=values)

    # Flatten MultiIndex columns
    # e.g., ('dist', -10) -> 'dist_t-10'
    df_wide.columns = [f"{col}_t{lag:+d}" for col, lag in df_wide.columns]

    df_wide = df_wide.reset_index()

    return df_wide


def _process_chunk(df_chunk, df_tracking):
    """
    Orchestrates the feature generation for a chunk of contact_ids.
    """
    # 1. Expand
    df_expanded = _expand_with_lags(df_chunk)

    # 2. Merge Tracking
    df_merged = _merge_window_tracking(df_expanded, df_tracking)

    # 3. Compute Dynamic Basis & Projections
    df_features = _compute_dynamic_basis_features(df_merged)

    # 4. Pivot
    df_wide = _pivot_to_wide(df_features)

    # 5. Scalars
    df_wide = _calculate_scalars(df_wide)

    # 6. Merge back targets/metadata (game_play, step, etc. are lost in pivot)
    # We merge back essential metadata from the original chunk
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ]
    if "contact" in df_chunk.columns:
        meta_cols.append("contact")

    df_wide = pd.merge(df_wide, df_chunk[meta_cols], on="contact_id", how="left")

    return df_wide


# =============================================================================
# MAIN GENERATION FUNCTIONS
# =============================================================================


def _generate_features_pipeline(metadata_df, tracking_path, desc="Generating Features"):
    """
    Generic pipeline to generate features from a metadata DataFrame and tracking file.
    """
    # 1. Load Tracking
    df_tracking = _load_full_tracking(tracking_path)

    # 2. Process in Chunks (by game_play to minimize tracking lookup, or just batches)
    # Grouping by game_play is efficient for tracking lookup if we were slicing,
    # but since we merged the whole tracking df (optimized), we can just batch rows.
    # Batch size: 5000 contact_ids
    BATCH_SIZE = 5000
    unique_ids = metadata_df["contact_id"].unique()

    # Split metadata into chunks
    chunks = [
        metadata_df[metadata_df["contact_id"].isin(batch)]
        for batch in np.array_split(unique_ids, np.ceil(len(unique_ids) / BATCH_SIZE))
    ]

    results = []
    print(f"Processing {len(chunks)} chunks...")

    for chunk in tqdm(chunks, desc=desc):
        # Filter tracking to relevant games in this chunk to speed up merge?
        # The merge function uses the full df_tracking.
        # For 1.2M rows tracking, it's fast enough.
        # If tracking is huge, we'd filter. Here it's okay.

        # Optimization: Filter tracking to only games in this chunk
        games_in_chunk = chunk["game_play"].unique()
        track_subset = df_tracking[df_tracking["game_play"].isin(games_in_chunk)]

        processed_chunk = _process_chunk(chunk, track_subset)
        results.append(processed_chunk)

    # Concatenate
    df_final = pd.concat(results, axis=0, ignore_index=True)

    # Memory cleanup
    del df_tracking
    gc.collect()

    return df_final


@utils.cache_result(file_format="parquet")
def generate_train_features(debug=False, sample_size=10000, load_cached_data=True):
    """
    Generates the training feature set.
    """
    # 1. Get Survivors from Gating Pipeline
    print("Retrieving gated training data...")
    df_meta = data_processing.process_train_data(
        debug=debug, sample_size=sample_size, load_cached_data=True
    )

    # 2. Generate Features
    return _generate_features_pipeline(
        df_meta, config.TRAIN_TRACKING_PATH, desc="Train Features"
    )


@utils.cache_result(file_format="parquet")
def generate_val_features(debug=False, sample_size=10000, load_cached_data=True):
    """
    Generates the validation feature set.
    """
    print("Retrieving gated validation data...")
    df_meta = data_processing.process_val_data(
        debug=debug, sample_size=sample_size, load_cached_data=True
    )

    return _generate_features_pipeline(
        df_meta, config.TRAIN_TRACKING_PATH, desc="Val Features"
    )


@utils.cache_result(file_format="parquet")
def generate_test_features(load_cached_data=True):
    """
    Generates the test feature set.
    """
    print("Retrieving test data...")
    # Note: process_test_data applies gating if configured.
    # Ensure config.GATING_THRESHOLD is safe for test.
    df_meta = data_processing.process_test_data(load_cached_data=True)

    return _generate_features_pipeline(
        df_meta, config.TEST_TRACKING_PATH, desc="Test Features"
    )
