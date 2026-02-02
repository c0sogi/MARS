import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from library import config, utils

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------


def _load_metadata(split):
    """Loads the appropriate metadata file based on the split."""
    if split == "train":
        return pd.read_csv(config.TRAIN_METADATA_PATH)
    elif split == "val":
        return pd.read_csv(config.VAL_METADATA_PATH)
    elif split == "test":
        return pd.read_csv(config.TEST_METADATA_PATH)
    else:
        raise ValueError(f"Unknown split: {split}")


def _load_tracking(split):
    """Loads the appropriate tracking data."""
    # Train and Val share the train_player_tracking.csv
    if split in ["train", "val"]:
        return pd.read_csv(config.TRAIN_TRACKING_PATH)
    elif split == "test":
        return pd.read_csv(config.TEST_TRACKING_PATH)
    else:
        raise ValueError(f"Unknown split: {split}")


def _compute_closing_speed(pos_a, vel_a, pos_b, vel_b):
    """
    Computes closing speed between two entities A and B.
    Closing Speed = - dot(RelVel, UnitRelPos)
    Positive value means closing in.
    """
    rel_pos = pos_b - pos_a
    rel_vel = vel_b - vel_a
    dist = np.linalg.norm(rel_pos, axis=-1, keepdims=True)

    # Avoid division by zero
    dist = np.maximum(dist, 1e-6)

    unit_rel_pos = rel_pos / dist
    # Dot product along the last dimension (x, y)
    closing_speed = -np.sum(rel_vel * unit_rel_pos, axis=-1)
    return closing_speed


# -----------------------------------------------------------------------------
# CORE PROCESSING LOGIC
# -----------------------------------------------------------------------------


def process_play(play_id, df_meta, df_track):
    """
    Processes a single play to generate features.

    Args:
        play_id: Unique ID of the play.
        df_meta: Metadata DataFrame for this play.
        df_track: Tracking DataFrame for this play.

    Returns:
        DataFrame containing features for the play.
    """
    # 1. Prepare Tracking Data Tensors
    # ---------------------------------------------------------
    # We need a dense representation: (Steps, Players, Features)
    # Map nfl_player_id to a 0..N index

    # Ensure IDs are integers for consistent mapping
    df_track["nfl_player_id"] = df_track["nfl_player_id"].fillna(-1).astype(int)

    unique_players = df_track["nfl_player_id"].unique()
    player_map = {pid: i for i, pid in enumerate(unique_players)}
    n_players = len(unique_players)

    # Steps need to be 0-indexed relative to the play or mapped
    # The 'step' column is already relative to play start.
    # We find the min and max step to define the tensor size
    min_step = df_track["step"].min()
    max_step = df_track["step"].max()
    n_steps = max_step - min_step + 1

    # Initialize tensors with NaNs
    # Shape: (T, N, 2) for pos, vel, acc
    pos_tensor = np.full((n_steps, n_players, 2), np.nan, dtype=np.float32)
    vel_tensor = np.full((n_steps, n_players, 2), np.nan, dtype=np.float32)
    acc_tensor = np.full((n_steps, n_players, 2), np.nan, dtype=np.float32)
    orient_tensor = np.full((n_steps, n_players), np.nan, dtype=np.float32)
    dir_tensor = np.full((n_steps, n_players), np.nan, dtype=np.float32)

    # Fill tensors
    # We iterate by player to fill slices (faster than row iteration)
    for pid, idx in player_map.items():
        p_track = df_track[df_track["nfl_player_id"] == pid]
        steps = p_track["step"].values - min_step

        # Safe indexing in case steps are out of bounds (unlikely given min/max)
        valid_mask = (steps >= 0) & (steps < n_steps)
        steps = steps[valid_mask]
        p_track = p_track.iloc[valid_mask]

        if len(steps) == 0:
            continue

        pos_tensor[steps, idx, 0] = p_track["x_position"].values
        pos_tensor[steps, idx, 1] = p_track["y_position"].values
        vel_tensor[steps, idx, 0] = p_track["speed"].values * np.sin(
            np.deg2rad(p_track["direction"].values)
        )
        vel_tensor[steps, idx, 1] = p_track["speed"].values * np.cos(
            np.deg2rad(p_track["direction"].values)
        )
        acc_tensor[steps, idx, 0] = p_track[
            "acceleration"
        ].values  # Approx direction? Use magnitude for now or re-derive
        # Note: 'acceleration' in csv is magnitude. 'sa' is signed.
        # We will store magnitude in channel 0 for simplicity or use raw columns if needed.
        # Let's store raw magnitude in 0.
        acc_tensor[steps, idx, 0] = p_track["acceleration"].values

        orient_tensor[steps, idx] = p_track["orientation"].values
        dir_tensor[steps, idx] = p_track["direction"].values

    # 2. Process Metadata (Vectorized Lookup)
    # ---------------------------------------------------------
    # Align metadata steps to tensor indices
    meta_steps = df_meta["step"].values - min_step

    # Filter metadata to valid steps
    valid_meta_mask = (meta_steps >= 0) & (meta_steps < n_steps)
    df_meta = df_meta[valid_meta_mask].copy()
    meta_steps = meta_steps[valid_meta_mask]

    # Identify P1 and P2 indices
    # Ensure P1 IDs are ints for lookup
    p1_ids = df_meta["nfl_player_id_1"].fillna(-1).astype(int).values
    p2_ids = df_meta["nfl_player_id_2"].values

    # Map IDs to tensor indices
    # Use -1 for missing/Ground
    p1_indices = np.array([player_map.get(pid, -1) for pid in p1_ids])

    # Handle Ground ('G') separately
    is_ground = p2_ids == "G"
    p2_indices = np.full(len(p2_ids), -1, dtype=int)

    # Only map non-ground P2s
    # Ensure non-ground P2 IDs are ints
    if np.any(~is_ground):
        non_ground_p2_ids = p2_ids[~is_ground]
        # Handle potential mixed types in p2_ids if not all G
        # It should be object array. Convert to numeric, coerce errors to -1
        # Convert to numeric, coercing errors to NaN (returns numpy array)
        numeric_ids = pd.to_numeric(non_ground_p2_ids, errors="coerce")
        # Use numpy function to fill NaNs, as .fillna is not available on ndarray
        non_ground_p2_ids = np.nan_to_num(numeric_ids, nan=-1).astype(int)

        p2_indices[~is_ground] = [player_map.get(pid, -1) for pid in non_ground_p2_ids]

    # Filter out rows where players are not in tracking data (rare data errors)
    valid_players_mask = (p1_indices != -1) & ((p2_indices != -1) | is_ground)
    if not np.any(valid_players_mask):
        return pd.DataFrame()

    df_meta = df_meta[valid_players_mask].reset_index(drop=True)
    meta_steps = meta_steps[valid_players_mask]
    p1_indices = p1_indices[valid_players_mask]
    p2_indices = p2_indices[valid_players_mask]
    is_ground = is_ground[valid_players_mask]

    n_samples = len(df_meta)

    # 3. Compute Basic Kinematics
    # ---------------------------------------------------------
    # Gather P1 data
    p1_pos = pos_tensor[meta_steps, p1_indices]  # (M, 2)
    p1_vel = vel_tensor[meta_steps, p1_indices]  # (M, 2)
    p1_acc = acc_tensor[meta_steps, p1_indices, 0]  # (M,)
    p1_dir = dir_tensor[meta_steps, p1_indices]
    p1_orient = orient_tensor[meta_steps, p1_indices]
    p1_speed = np.linalg.norm(p1_vel, axis=1)

    # Gather P2 data (Handle Ground)
    p2_pos = np.zeros((n_samples, 2), dtype=np.float32)
    p2_vel = np.zeros((n_samples, 2), dtype=np.float32)
    p2_acc = np.zeros((n_samples,), dtype=np.float32)
    p2_dir = np.zeros((n_samples,), dtype=np.float32)
    p2_orient = np.zeros((n_samples,), dtype=np.float32)
    p2_speed = np.zeros((n_samples,), dtype=np.float32)

    # Fill for non-ground
    not_g = ~is_ground
    if np.any(not_g):
        idx_ng = p2_indices[not_g]
        steps_ng = meta_steps[not_g]
        p2_pos[not_g] = pos_tensor[steps_ng, idx_ng]
        p2_vel[not_g] = vel_tensor[steps_ng, idx_ng]
        p2_acc[not_g] = acc_tensor[steps_ng, idx_ng, 0]
        p2_dir[not_g] = dir_tensor[steps_ng, idx_ng]
        p2_orient[not_g] = orient_tensor[steps_ng, idx_ng]
        p2_speed[not_g] = np.linalg.norm(p2_vel[not_g], axis=1)

    # Calculate Distance
    # Sentinel Strategy: -1.0 for Ground
    dist = np.full(n_samples, config.GROUND_SENTINEL, dtype=np.float32)
    if np.any(not_g):
        diff = p1_pos[not_g] - p2_pos[not_g]
        dist[not_g] = np.linalg.norm(diff, axis=1)

    # 4. Compute Invariant Extremum-Context Features
    # ---------------------------------------------------------
    # We need to scan all players at the relevant steps
    # Shape of All_Pos at relevant steps: (M, N_players, 2)
    all_pos = pos_tensor[meta_steps]  # (M, N, 2)
    all_vel = vel_tensor[meta_steps]  # (M, N, 2)
    all_acc = acc_tensor[meta_steps, :, 0]  # (M, N)

    # Calculate Distances to P1 and P2
    # Expand P1/P2 to (M, 1, 2) for broadcasting
    dist_to_p1 = np.linalg.norm(all_pos - p1_pos[:, None, :], axis=2)  # (M, N)

    dist_to_p2 = np.full((n_samples, n_players), np.inf, dtype=np.float32)
    if np.any(not_g):
        dist_to_p2[not_g] = np.linalg.norm(
            all_pos[not_g] - p2_pos[not_g][:, None, :], axis=2
        )

    # Minimum distance to the pair (or just P1 if Ground)
    # For Ground, dist_to_p2 is inf, so min is dist_to_p1, which is correct.
    dist_to_pair = np.minimum(dist_to_p1, dist_to_p2)  # (M, N)

    # Mask self-interactions (P1 and P2 shouldn't be counted as 3rd party)
    # Create a mask (M, N)
    row_indices = np.arange(n_samples)
    mask = np.zeros((n_samples, n_players), dtype=bool)
    mask[row_indices, p1_indices] = True

    # For non-ground, mask P2
    if np.any(not_g):
        # We need to handle the fact that p2_indices has -1 for ground
        # Only set mask where not ground
        mask[np.where(not_g)[0], p2_indices[not_g]] = True

    # Apply mask (set distance to inf)
    dist_to_pair[mask] = np.inf

    # Extremum 1: Min Dist to 3rd Party
    min_dist_3rd = np.min(dist_to_pair, axis=1)

    # Extremum 2: Max Closing Speed of 3rd Party
    # Identify neighbors within context radius
    neighbor_mask = (dist_to_pair < config.CONTEXT_RADIUS) & (~mask)

    # Compute closing speed to P1 and P2 for all players
    # Closing speed P3->P1
    cs_p1 = _compute_closing_speed(
        p1_pos[:, None, :], p1_vel[:, None, :], all_pos, all_vel
    )

    # Closing speed P3->P2 (default -inf)
    cs_p2 = np.full((n_samples, n_players), -np.inf, dtype=np.float32)
    if np.any(not_g):
        cs_p2[not_g] = _compute_closing_speed(
            p2_pos[not_g][:, None, :],
            p2_vel[not_g][:, None, :],
            all_pos[not_g],
            all_vel[not_g],
        )

    # Max closing speed to either player in the pair
    cs_pair = np.maximum(cs_p1, cs_p2)

    # Filter by neighbor mask (only consider close players)
    # If no neighbors, we want a low value (e.g., -10)
    cs_pair[~neighbor_mask] = -10.0
    max_closing_speed_3rd = np.max(cs_pair, axis=1)

    # Extremum 3: Max Acceleration of 3rd Party
    acc_pair = all_acc.copy()
    acc_pair[~neighbor_mask] = -1.0
    max_acc_3rd = np.max(acc_pair, axis=1)

    # 5. Assemble DataFrame
    # ---------------------------------------------------------
    features = pd.DataFrame(
        {
            "distance": dist,
            "speed_p1": p1_speed,
            "speed_p2": p2_speed,
            "acceleration_p1": p1_acc,
            "acceleration_p2": p2_acc,
            "direction_p1": p1_dir,
            "direction_p2": p2_dir,
            "orientation_p1": p1_orient,
            "orientation_p2": p2_orient,
            "x_position_p1": p1_pos[:, 0],
            "y_position_p1": p1_pos[:, 1],
            "x_position_p2": p2_pos[:, 0],
            "y_position_p2": p2_pos[:, 1],
            "min_dist_3rd_party": min_dist_3rd,
            "max_closing_speed_3rd_party": max_closing_speed_3rd,
            "max_acceleration_3rd_party": max_acc_3rd,
        }
    )

    # Derived features
    features["speed_diff"] = np.abs(features["speed_p1"] - features["speed_p2"])
    features["acc_diff"] = np.abs(
        features["acceleration_p1"] - features["acceleration_p2"]
    )
    features["direction_diff"] = np.abs(
        features["direction_p1"] - features["direction_p2"]
    )
    features["orientation_diff"] = np.abs(
        features["orientation_p1"] - features["orientation_p2"]
    )

    # Add metadata columns back for grouping/windowing
    features["game_play"] = df_meta["game_play"]
    features["contact_id"] = df_meta["contact_id"]
    features["step"] = df_meta["step"]
    features["nfl_player_id_1"] = df_meta["nfl_player_id_1"]
    features["nfl_player_id_2"] = df_meta["nfl_player_id_2"]
    features["contact"] = df_meta["contact"]

    return features


def apply_temporal_windowing(df):
    """
    Applies rolling window to flatten temporal features.
    """
    # Sort to ensure time order
    df = df.sort_values(by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

    # Define features to lag
    cols_to_lag = config.TEMPORAL_TARGET_FEATURES

    # Group by pair
    # Note: nfl_player_id_2 can be 'G' (str) or int. Convert to str for consistent grouping.
    df["p2_str"] = df["nfl_player_id_2"].astype(str)
    grouper = df.groupby(["game_play", "nfl_player_id_1", "p2_str"])

    # Generate lags
    new_cols = {}
    for col in cols_to_lag:
        if col not in df.columns:
            continue

        for lag in range(1, config.WINDOW_SIZE + 1):
            # Lag (Past)
            new_cols[f"{col}_lag_{lag}"] = grouper[col].shift(lag)
            # Lead (Future)
            new_cols[f"{col}_lead_{lag}"] = grouper[col].shift(-lag)

    # Concatenate new columns
    df_lags = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, df_lags], axis=1)

    # Fill NaNs (edges of play)
    # We use ffill/bfill within the group logic, but doing it on the whole sorted df
    # with limit might bleed across pairs.
    # Safer: Fill with current value (0-order hold) or 0.
    # Given the constraints, simple fillna(0) for lags is common,
    # but forward filling the instantaneous value is better physics.
    # Let's use bfill/ffill grouped.
    for col in df_lags.columns:
        df[col] = df[col].fillna(method="bfill").fillna(method="ffill")

    df = df.drop(columns=["p2_str"])
    return df


def apply_geometric_gating(df):
    """
    Filters the dataset based on distance and sentinel strategy.
    """
    # Keep if Ground Contact OR Distance < Threshold
    # Ground contacts have distance == -1.0
    mask = (df["nfl_player_id_2"] == "G") | (df["distance"] <= config.GATING_THRESHOLD)
    return df[mask].reset_index(drop=True)


# -----------------------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------------------


def generate_features(split: str, load_cached: bool = True):
    """
    Main function to generate features for a given split.
    Handles caching, processing, windowing, and gating.
    """
    cache_file = os.path.join(config.WORKING_DIR, f"features_{split}.parquet")

    # 1. Check Cache
    if load_cached and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating features for {split}...")

    # 2. Load Data
    df_meta = _load_metadata(split)
    df_track = _load_tracking(split)

    # Ensure game_play is string in both to ensure matching
    df_meta["game_play"] = df_meta["game_play"].astype(str)
    if "game_play" in df_track.columns:
        df_track["game_play"] = df_track["game_play"].astype(str)

    # 3. Process Play-by-Play
    unique_plays = df_meta["game_play"].unique()
    processed_dfs = []

    # Use tqdm for progress tracking
    for play_id in tqdm(unique_plays, desc=f"Processing {split} plays"):
        # Subset data
        play_meta = df_meta[df_meta["game_play"] == play_id].copy()
        play_track = df_track[df_track["game_play"] == play_id].copy()

        if play_meta.empty or play_track.empty:
            continue

        # Compute Instantaneous Features (including Context)
        df_play = process_play(play_id, play_meta, play_track)

        if not df_play.empty:
            processed_dfs.append(df_play)

    if not processed_dfs:
        print("Warning: No features generated.")
        return pd.DataFrame()

    # Concatenate all plays
    df_full = pd.concat(processed_dfs, axis=0, ignore_index=True)

    # Free memory
    del df_meta, df_track, processed_dfs
    gc.collect()

    # 4. Temporal Windowing
    print("Applying temporal windowing...")
    df_full = apply_temporal_windowing(df_full)

    # 5. Geometric Gating
    # Only apply gating to Train/Val. Test set must keep all rows for submission.
    if split != "test":
        print(
            f"Applying geometric gating (Threshold: {config.GATING_THRESHOLD} yds)..."
        )
        before_len = len(df_full)
        df_full = apply_geometric_gating(df_full)
        print(f"Rows reduced from {before_len} to {len(df_full)}")

    # 6. Memory Optimization
    df_full = utils.reduce_mem_usage(df_full)

    # 7. Save to Cache
    print(f"Saving features to {cache_file}...")
    df_full.to_parquet(cache_file, index=False)

    return df_full
