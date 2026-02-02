import os
import gc
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from library.config import (
    WORKING_DIR,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    CACHE_TRAIN_FEATURES,
    CACHE_VAL_FEATURES,
    CACHE_TEST_FEATURES,
    GATING_DISTANCE,
    FLOW_NEIGHBOR_RADIUS,
    WINDOW_SIZE,
    RANDOM_STATE,
)
from library.data_loader import load_metadata, load_tracking
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(RANDOM_STATE)


def compute_physics_derivatives(df):
    """
    Computes Jerk and Angular Jerk for the tracking data.
    Assumes df is sorted by game_play, nfl_player_id, step.
    """
    # Ensure sorted order for temporal diffs
    df = df.sort_values(by=["game_play", "nfl_player_id", "step"]).reset_index(
        drop=True
    )

    # Group by player within play to prevent boundary bleeding between plays
    g = df.groupby(["game_play", "nfl_player_id"])

    # Jerk: Derivative of acceleration (magnitude)
    # We fill the first value with 0
    df["jerk"] = g["acceleration"].diff().fillna(0)

    # Angular Jerk: Derivative of orientation
    # We must handle the circular nature of orientation (0-360 degrees)
    orient = df["orientation"].values
    # Shift using pandas to respect groups
    prev_orient = (
        g["orientation"].shift(1).fillna(orient[0])
    )  # Fill with self -> diff 0

    # Circular difference: shortest path between angles
    diff = (orient - prev_orient + 180) % 360 - 180
    df["angular_jerk"] = diff

    return df


def compute_flow_context(df):
    """
    Computes Radial and Tangential Flux for each player at each step.
    Uses vectorized numpy operations per play to handle the O(N^2) complexity efficiently.
    """
    # 1. Convert velocity to Cartesian components for vector math
    # NFL Standard: 0 deg is Y-axis (North), 90 deg is X-axis (East)
    # vx = speed * sin(rad), vy = speed * cos(rad)
    rads = np.deg2rad(df["direction"].values)
    df["vx"] = df["speed"] * np.sin(rads)
    df["vy"] = df["speed"] * np.cos(rads)

    # Prepare result arrays initialized to 0
    radial_flux = np.zeros(len(df), dtype=np.float32)
    tangential_flux = np.zeros(len(df), dtype=np.float32)

    # Map original indices to put results back correctly
    df["orig_idx"] = np.arange(len(df))

    # Process per play to keep memory usage reasonable (Batch Processing)
    play_groups = df.groupby("game_play")

    for play_id, play_df in play_groups:
        # Extract numpy arrays for the play
        steps = play_df["step"].values
        coords = play_df[["x_position", "y_position"]].values
        vels = play_df[["vx", "vy"]].values
        indices = play_df["orig_idx"].values

        # Iterate over unique steps in this play
        unique_steps = np.unique(steps)

        for t in unique_steps:
            # Mask for current step
            mask = steps == t
            if not np.any(mask):
                continue

            step_indices = indices[mask]
            step_coords = coords[mask]
            step_vels = vels[mask]

            n_players = len(step_coords)
            if n_players < 2:
                continue

            # --- Vectorized Flow Calculation ---

            # 1. Distance Matrix (N, N)
            # d_vec[i, j] = vector from i to j
            d_vec = step_coords[:, None, :] - step_coords[None, :, :]  # Shape (N, N, 2)
            d_norm = np.linalg.norm(d_vec, axis=2)  # Shape (N, N)

            # Avoid divide by zero on diagonal
            d_norm[d_norm == 0] = 1e-6

            # 2. Neighbor Mask
            # Distance < Radius AND not self
            neighbor_mask = (d_norm < FLOW_NEIGHBOR_RADIUS) & (d_norm > 1e-5)

            # 3. Relative Velocity
            # v_rel[i, j] = v_neighbor(j) - v_target(i)
            v_rel = step_vels[None, :, :] - step_vels[:, None, :]  # Shape (N, N, 2)

            # 4. Normalized Direction Vector (Target -> Neighbor)
            r_hat = d_vec / d_norm[:, :, None]

            # 5. Radial Component: v_rel dot r_hat
            # Project relative velocity onto the connecting line
            # Negative = Converging, Positive = Diverging
            radial_comp = np.sum(v_rel * r_hat, axis=2)

            # 6. Tangential Component: v_rel dot r_perp
            # r_perp is r_hat rotated 90 degrees: (-y, x)
            r_perp = np.stack([-r_hat[:, :, 1], r_hat[:, :, 0]], axis=2)
            tangential_comp = np.sum(v_rel * r_perp, axis=2)

            # 7. Aggregate
            # Zero out non-neighbors
            radial_comp[~neighbor_mask] = 0
            tangential_comp[~neighbor_mask] = 0

            neighbor_counts = neighbor_mask.sum(axis=1)

            # Compute means where counts > 0
            valid_mask = neighbor_counts > 0

            sum_radial = radial_comp.sum(axis=1)
            sum_tangential = tangential_comp.sum(axis=1)

            curr_radial = np.zeros(n_players)
            curr_tangential = np.zeros(n_players)

            curr_radial[valid_mask] = (
                sum_radial[valid_mask] / neighbor_counts[valid_mask]
            )
            curr_tangential[valid_mask] = (
                sum_tangential[valid_mask] / neighbor_counts[valid_mask]
            )

            # Store results
            radial_flux[step_indices] = curr_radial
            tangential_flux[step_indices] = curr_tangential

    df["radial_flux"] = radial_flux
    df["tangential_flux"] = tangential_flux

    # Cleanup temporary columns
    df = df.drop(columns=["vx", "vy", "orig_idx"])
    return df


def create_lag_features(df):
    """
    Creates flattened temporal window features for the tracking data.
    Generates columns like 'speed_t-10', 'speed_t0', 'speed_t+10'.
    """
    # Columns to create lags for
    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "jerk",
        "angular_jerk",
        "radial_flux",
        "tangential_flux",
    ]

    # Sort is critical for shift
    df = df.sort_values(by=["game_play", "nfl_player_id", "step"])

    g = df.groupby(["game_play", "nfl_player_id"])

    # Define window range
    shifts = range(-WINDOW_SIZE, WINDOW_SIZE + 1)

    new_cols = {}

    for col in feature_cols:
        for s in shifts:
            # Naming convention: {col}_t{shift}
            col_name = f"{col}_t{s}"

            if s == 0:
                new_cols[col_name] = df[col]
            else:
                # pandas shift(s) shifts data down by s.
                # To get value at t+s (future) into row t, we shift UP by s (shift(-s)).
                # To get value at t-s (past) into row t, we shift DOWN by s (shift(s)).
                # Since s ranges from -10 to +10:
                # If s = -10 (past), we want value from t-10. shift(10).
                # If s = +10 (future), we want value from t+10. shift(-10).
                # So we shift by -s.
                new_cols[col_name] = g[col].shift(-s)

    # Create DataFrame from new columns
    df_lags = pd.DataFrame(new_cols, index=df.index)

    # Concatenate with key columns
    df_final = pd.concat([df[["game_play", "nfl_player_id", "step"]], df_lags], axis=1)

    return df_final


def process_tracking_data(split, load_cached_data=True, nrows=None):
    """
    Orchestrates the processing of tracking data: Physics -> Flow -> Lags.
    Handles caching based on source file (train vs test).
    """
    # Determine cache file path based on source
    # Train and Val share the same tracking file
    source_name = "train" if split in ["train", "val"] else "test"
    cache_path = os.path.join(WORKING_DIR, f"processed_tracking_{source_name}.parquet")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed tracking data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing tracking data for {source_name} (fresh)...")

    # Load raw tracking
    df = load_tracking(split, nrows=nrows, load_cached_data=load_cached_data)

    # 1. Compute Physics Derivatives
    df = compute_physics_derivatives(df)

    # 2. Compute Flow Context
    df = compute_flow_context(df)

    # 3. Create Temporal Lags
    df = create_lag_features(df)

    # Cache result (only if full dataset)
    if nrows is None:
        print(f"Caching processed tracking data to {cache_path}...")
        df.to_parquet(cache_path, index=False)

    return df


def generate_features(split, load_cached_data=True, nrows=None, gating=True):
    """
    Main pipeline to generate the final feature set for a split.
    Merges processed tracking data onto the contact metadata.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Use cached files if available.
        nrows (int): Limit rows for debugging.
        gating (bool): Whether to apply Geometric Gating (distance filter).
                       Should be True for Train/Val, False for Test.
    """
    # Determine cache path
    cache_map = {
        "train": CACHE_TRAIN_FEATURES,
        "val": CACHE_VAL_FEATURES,
        "test": CACHE_TEST_FEATURES,
    }
    cache_path = cache_map[split]

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {split}...")

    # 1. Load Metadata
    df_meta = load_metadata(split, nrows=nrows, load_cached_data=load_cached_data)

    # 2. Load Processed Tracking
    df_track = process_tracking_data(
        split, load_cached_data=load_cached_data, nrows=nrows
    )

    # 3. Geometric Gating (The Sieve)
    if gating:
        print(f"Applying Geometric Gating (Dist < {GATING_DISTANCE}y)...")

        # We need distance at t=0 to filter.
        # Extract t0 coordinates for efficient merge check
        cols_t0 = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position_t0",
            "y_position_t0",
        ]
        df_t0 = df_track[cols_t0].copy()

        # Merge P1 coordinates
        df_gate = df_meta[
            ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
        ].copy()
        df_gate = (
            df_gate.merge(
                df_t0,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            .rename(columns={"x_position_t0": "x1", "y_position_t0": "y1"})
            .drop(columns=["nfl_player_id"])
        )

        # Merge P2 coordinates (handle 'G')
        # Create numeric key for P2, coercing 'G' to NaN
        df_gate["p2_int"] = pd.to_numeric(df_gate["nfl_player_id_2"], errors="coerce")
        mask_g = df_gate["nfl_player_id_2"] == "G"

        df_gate = (
            df_gate.merge(
                df_t0,
                left_on=["game_play", "step", "p2_int"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            .rename(columns={"x_position_t0": "x2", "y_position_t0": "y2"})
            .drop(columns=["nfl_player_id", "p2_int"])
        )

        # Calculate Euclidean Distance
        d = np.sqrt(
            (df_gate["x1"] - df_gate["x2"]) ** 2 + (df_gate["y1"] - df_gate["y2"]) ** 2
        )

        # Keep if Distance < Threshold OR P2 is Ground
        # Note: If tracking data is missing (NaN dist), we keep to be safe?
        # Or drop? Usually drop if no tracking. But let's assume valid tracking.
        keep_mask = (d <= GATING_DISTANCE) | mask_g | d.isna()

        before_len = len(df_meta)
        df_meta = df_meta[keep_mask].reset_index(drop=True)
        print(f"Gating reduced samples from {before_len} to {len(df_meta)}.")

        # Cleanup
        del df_gate, df_t0, d
        gc.collect()

    # 4. Full Feature Merge
    print("Merging full temporal features...")

    # Identify feature columns (exclude keys)
    track_cols = [
        c for c in df_track.columns if c not in ["game_play", "step", "nfl_player_id"]
    ]

    # Merge P1 Features
    df_features = df_meta.merge(
        df_track,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns
    rename_p1 = {c: f"p1_{c}" for c in track_cols}
    df_features = df_features.rename(columns=rename_p1)

    # Merge P2 Features
    # Create temp join key for P2
    df_features["p2_join_key"] = pd.to_numeric(
        df_features["nfl_player_id_2"], errors="coerce"
    )

    df_features = df_features.merge(
        df_track,
        left_on=["game_play", "step", "p2_join_key"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    ).drop(columns=["nfl_player_id", "p2_join_key"])

    # Rename P2 columns
    rename_p2 = {c: f"p2_{c}" for c in track_cols}
    df_features = df_features.rename(columns=rename_p2)

    # Fill NaNs for Ground or missing tracking
    # For Ground, all P2 features (speed, flux, etc.) become 0.
    all_feat_cols = list(rename_p1.values()) + list(rename_p2.values())
    df_features[all_feat_cols] = df_features[all_feat_cols].fillna(0)

    # 5. Calculate Relative Features at t=0
    print("Calculating relative interaction features...")

    # Distance
    p1_x = df_features["p1_x_position_t0"]
    p1_y = df_features["p1_y_position_t0"]
    p2_x = df_features["p2_x_position_t0"]
    p2_y = df_features["p2_y_position_t0"]

    df_features["distance"] = np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

    # Speed Diff
    p1_s = df_features["p1_speed_t0"]
    p2_s = df_features["p2_speed_t0"]
    df_features["speed_diff"] = np.abs(p1_s - p2_s)

    # Ground Flag
    df_features["is_ground"] = (df_features["nfl_player_id_2"] == "G").astype(int)

    # Handle Ground Distance (Set to 0 for logic consistency, though P2 coords are 0)
    df_features.loc[df_features["is_ground"] == 1, "distance"] = 0

    # Cache result
    if nrows is None:
        print(f"Caching features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

    return df_features
