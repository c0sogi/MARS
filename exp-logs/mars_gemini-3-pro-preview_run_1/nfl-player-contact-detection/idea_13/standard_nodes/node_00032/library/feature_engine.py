import os
import gc
import pandas as pd
import numpy as np
from library.config import WORKING_DIR, IKS_NEIGHBOR_RADIUS, WINDOW_SIZE, SEED
from library.utils import reduce_mem_usage, generate_cache_key
from library.data_loader import load_tracking


def compute_derived_tracking_features(df_tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics derivatives (Jerk) and vector components from raw tracking data.
    """
    df = df_tracking.copy()

    # Ensure sorted for time-based diffs
    df = df.sort_values(["game_play", "nfl_player_id", "step"])

    # Convert direction to radians (0 is North/Y, 90 is East/X in standard NFL tracking)
    df["dir_rad"] = np.deg2rad(df["direction"])

    # Velocity components
    df["v_x"] = df["speed"] * np.sin(df["dir_rad"])
    df["v_y"] = df["speed"] * np.cos(df["dir_rad"])

    # Jerk: Derivative of acceleration magnitude
    # Group by player to ensure we don't diff across different players
    df["jerk"] = (
        df.groupby(["game_play", "nfl_player_id"])["acceleration"].diff().fillna(0)
    )

    return df


def create_temporal_features(df_tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Creates flattened temporal windows (+/- WINDOW_SIZE) for tracking features.
    Returns a 'wide' dataframe with columns like speed_prev_1, speed_next_1, etc.
    """
    # Features to window
    cols_to_window = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "jerk",
        "direction",
        "orientation",
    ]

    # Start with keys
    df_res = df_tracking[["game_play", "nfl_player_id", "step"]].copy()

    # Group object for efficient shifting
    grouped = df_tracking.groupby(["game_play", "nfl_player_id"])

    # Loop through lags
    for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        if lag == 0:
            continue

        suffix = f"_prev_{-lag}" if lag < 0 else f"_next_{lag}"

        # Shift: shift(-1) gives the value at t+1 (next row)
        shifted = grouped[cols_to_window].shift(-lag)

        # Rename columns
        shifted.columns = [f"{c}{suffix}" for c in cols_to_window]

        # Concatenate
        df_res = pd.concat([df_res, shifted], axis=1)

    return df_res


def create_iks_features(
    df_base: pd.DataFrame, df_tracking: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes Invariant Kinematic Set (IKS) features.
    Aggregates relative physics of neighbors within radius.
    """
    # 1. Prepare Context Data
    # Filter tracking to relevant plays/steps to optimize merge
    relevant_keys = df_base[["game_play", "step"]].drop_duplicates()
    df_ctx = pd.merge(df_tracking, relevant_keys, on=["game_play", "step"], how="inner")

    # Select and rename context columns
    ctx_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "direction",
        "v_x",
        "v_y",
    ]
    df_ctx = df_ctx[ctx_cols].rename(
        columns={
            "nfl_player_id": "ctx_id",
            "x_position": "ctx_x",
            "y_position": "ctx_y",
            "speed": "ctx_s",
            "direction": "ctx_dir",
            "v_x": "ctx_vx",
            "v_y": "ctx_vy",
        }
    )

    # 2. Merge Context onto Base
    # We use a temp row_id to aggregate back later
    df_work = df_base.copy()
    if "row_id" not in df_work.columns:
        df_work["row_id"] = np.arange(len(df_work))

    # Inner join expands rows: (Target Pairs) x (All Players in Step)
    merged = pd.merge(df_work, df_ctx, on=["game_play", "step"], how="inner")

    # 3. Filter Neighbors (Remove Self and Partner)
    merged["p1_str"] = merged["nfl_player_id_1"].astype(str)
    merged["p2_str"] = merged["nfl_player_id_2"].astype(str)
    merged["ctx_str"] = merged["ctx_id"].astype(str)

    mask_not_p1 = merged["ctx_str"] != merged["p1_str"]
    mask_not_p2 = merged["ctx_str"] != merged["p2_str"]
    merged = merged[mask_not_p1 & mask_not_p2]

    # 4. Calculate Distances
    dx1 = merged["ctx_x"] - merged["x_position_p1"]
    dy1 = merged["ctx_y"] - merged["y_position_p1"]
    dist1 = np.sqrt(dx1**2 + dy1**2)

    # Handle Ground (P2='G')
    is_ground = merged["p2_str"] == "G"
    dx2 = merged["ctx_x"] - merged["x_position_p2"]
    dy2 = merged["ctx_y"] - merged["y_position_p2"]
    dist2 = np.sqrt(dx2**2 + dy2**2)
    # If P2 is Ground, distance to P2 is irrelevant/infinite for neighbor checking
    dist2 = dist2.where(~is_ground, np.inf)

    # 5. Apply Radius Filter (The Sieve)
    mask_close = (dist1 <= IKS_NEIGHBOR_RADIUS) | (dist2 <= IKS_NEIGHBOR_RADIUS)
    survivors = merged[mask_close].copy()

    # 6. Compute Relative Kinematics (Vectorized)
    # Relative Speed to P1
    survivors["rel_speed_p1"] = np.abs(survivors["ctx_s"] - survivors["speed_p1"])

    # Closing Speed to P1: Projection of relative velocity onto position vector
    # v_rel = v_ctx - v_p1
    # r_rel = r_ctx - r_p1
    # closing_spd = -(v_rel . r_rel) / |r_rel|

    # Re-calculate P1 velocity components (incase not in df_base)
    # We assume df_base has speed_p1, direction_p1 from data loader merge
    p1_vx = survivors["speed_p1"] * np.sin(np.deg2rad(survivors["direction_p1"]))
    p1_vy = survivors["speed_p1"] * np.cos(np.deg2rad(survivors["direction_p1"]))

    dvx1 = survivors["ctx_vx"] - p1_vx
    dvy1 = survivors["ctx_vy"] - p1_vy

    # Dot product
    dot_p1 = dvx1 * dx1[survivors.index] + dvy1 * dy1[survivors.index]

    # Safe divide
    d1_safe = dist1[survivors.index].replace(0, 0.1)
    survivors["closing_speed_p1"] = -(dot_p1 / d1_safe)

    # Map distance back for aggregation
    survivors["dist_to_p1"] = dist1[survivors.index]

    # 7. Aggregation
    aggs = {
        "dist_to_p1": ["min", "mean"],
        "rel_speed_p1": ["max", "mean"],
        "closing_speed_p1": ["max", "mean"],
    }

    grouped = survivors.groupby("row_id").agg(aggs)

    # Flatten MultiIndex columns
    grouped.columns = [f"iks_{c[0]}_{c[1]}" for c in grouped.columns]

    # 8. Merge back to ensure all rows exist (fill NaNs for isolated pairs)
    df_res = df_work[["row_id"]].merge(grouped, on="row_id", how="left")

    # Fill NaNs with physical logic
    fill_vals = {
        "iks_dist_to_p1_min": 10.0,  # No neighbors -> far away
        "iks_dist_to_p1_mean": 10.0,
        "iks_rel_speed_p1_max": 0.0,  # No relative motion
        "iks_rel_speed_p1_mean": 0.0,
        "iks_closing_speed_p1_max": -10.0,  # Not closing in
        "iks_closing_speed_p1_mean": 0.0,
    }

    for col, val in fill_vals.items():
        if col in df_res.columns:
            df_res[col] = df_res[col].fillna(val)

    return df_res


def generate_features(
    df: pd.DataFrame, split: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Main pipeline to generate all features (Temporal + IKS).
    """
    # 0. Cache Check
    cache_params = {
        "split": split,
        "rows": len(df),
        "window": WINDOW_SIZE,
        "iks_rad": IKS_NEIGHBOR_RADIUS,
    }
    cache_key = generate_cache_key(cache_params)
    cache_path = os.path.join(WORKING_DIR, f"features_{split}_{cache_key}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating features for {split} ({len(df)} rows)...")

    # 1. Load and Process Tracking
    print("Loading and processing tracking data...")
    df_tracking = load_tracking(split)

    # Filter tracking to relevant plays to save memory/time
    relevant_plays = df["game_play"].unique()
    df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

    # Compute Derivatives (Jerk, Vectors)
    df_tracking = compute_derived_tracking_features(df_tracking)

    # 2. Temporal Features (Windowing)
    print("Creating flattened temporal windows...")
    df_temporal = create_temporal_features(df_tracking)

    # Merge Temporal Features for Player 1
    print("Merging temporal features for P1...")
    df = df.merge(
        df_temporal,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns
    temp_cols = [
        c
        for c in df_temporal.columns
        if c not in ["game_play", "nfl_player_id", "step"]
    ]
    rename_p1 = {c: f"{c}_p1" for c in temp_cols}
    df = df.rename(columns=rename_p1)

    # Merge Temporal Features for Player 2
    print("Merging temporal features for P2...")
    # Create numeric join key for P2 (handling 'G')
    df["p2_join"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

    df = df.merge(
        df_temporal,
        left_on=["game_play", "p2_join", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id", "p2_join"])

    # Rename P2 columns
    rename_p2 = {c: f"{c}_p2" for c in temp_cols}
    df = df.rename(columns=rename_p2)

    # 3. Invariant Kinematic Set (IKS) Features
    print("Computing Invariant Kinematic Set (IKS) features...")
    # Add temporary row_id for safe merging
    df["row_id"] = np.arange(len(df))

    df_iks = create_iks_features(df, df_tracking)

    # Merge IKS features back
    df = df.merge(df_iks, on="row_id", how="left")
    df = df.drop(columns=["row_id"])

    # 4. Final Cleanup
    df = reduce_mem_usage(df)

    # Save to cache
    print(f"Saving features to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    # Clean up memory
    del df_tracking, df_temporal, df_iks
    gc.collect()

    return df
