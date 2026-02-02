import pandas as pd
import numpy as np
import os
import glob
from library.config import Config
from library.utils import (
    calculate_shortest_arc,
    calculate_euclidean_distance,
    calculate_log_distance,
    calculate_closing_speed,
)

# Hardcoded list of NFL positions to ensure consistent encoding across Train/Test
POSITIONS_VOCAB = [
    "QB",
    "WR",
    "TE",
    "RB",
    "FB",
    "T",
    "G",
    "C",
    "DE",
    "DT",
    "NT",
    "LB",
    "ILB",
    "OLB",
    "MLB",
    "CB",
    "S",
    "SS",
    "FS",
    "DB",
    "K",
    "P",
    "LS",
]
POS_TO_IDX = {pos: i for i, pos in enumerate(POSITIONS_VOCAB)}
TEAM_TO_IDX = {"home": 0, "away": 1}


def get_pos_idx(pos_series):
    return pos_series.map(POS_TO_IDX).fillna(len(POSITIONS_VOCAB)).astype(int)


def get_team_idx(team_series):
    return team_series.map(TEAM_TO_IDX).fillna(-1).astype(int)


def process_tracking_data(filepath, game_plays):
    """
    Loads tracking data, calculates velocities, and generates temporal window features.
    Returns a wide DataFrame with lagged columns.
    """
    # Load only necessary columns to save memory
    cols = [
        "game_play",
        "nfl_player_id",
        "step",
        "position",
        "team",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
        "distance",
    ]

    # Read csv
    df = pd.read_csv(filepath, usecols=lambda c: c in cols)

    # Filter for relevant game_plays
    df = df[df["game_play"].isin(game_plays)].copy()

    # Calculate Velocity Components (0 deg is Y-axis, 90 deg is X-axis convention)
    # Convert to radians
    dir_rad = np.deg2rad(df["direction"].fillna(0))
    df["vx"] = df["speed"] * np.sin(dir_rad)
    df["vy"] = df["speed"] * np.cos(dir_rad)

    # Calculate Orientation Components
    ori_rad = np.deg2rad(df["orientation"].fillna(0))
    df["orientation_sin"] = np.sin(ori_rad)
    df["orientation_cos"] = np.cos(ori_rad)

    # Sort for shifting
    df.sort_values(["game_play", "nfl_player_id", "step"], inplace=True)

    # Features to window
    # Note: 'distance' in raw data is distance traveled. Renaming to avoid conflict with interaction distance.
    df.rename(columns={"distance": "distance_traveled"}, inplace=True)

    feats_to_lag = Config.KINEMATIC_FEATURES

    # Generate Lags
    # We want lags from -W to +W
    # Group by player to ensure shifts don't cross players
    grouped = df.groupby(["game_play", "nfl_player_id"])

    lagged_dfs = []
    # Include the base columns (metadata)
    base_cols = ["game_play", "nfl_player_id", "step", "position", "team"]
    lagged_dfs.append(df[base_cols])

    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        # shift(k) shifts data down by k.
        # To get t-k (past), we shift by k.
        # To get t+k (future), we shift by -k.
        # Let's stick to convention: lag k means value at t+k relative to t.
        # So we use shift(-lag).
        shifted = grouped[feats_to_lag].shift(-lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in feats_to_lag]
        lagged_dfs.append(shifted)

    df_wide = pd.concat(lagged_dfs, axis=1)

    # Fill NaNs created by shifting (edges of play) with appropriate values?
    # For now, we leave them as NaN or fill with 0.
    # Neural nets don't like NaNs. Forward fill/Back fill is better for time series.
    # However, grouping makes ffill hard on wide format.
    # We will fill with 0 for now as padding.
    df_wide = df_wide.fillna(0)

    return df_wide


def process_visual_data(filepath, game_plays):
    """
    Loads helmet data, maps frames to steps, applies Max-Pooling,
    and generates temporal window features.
    """
    df = pd.read_csv(filepath)
    df = df[df["game_play"].isin(game_plays)].copy()

    # Map Frame to Step
    # Heuristic: Step 0 is approx Frame 300. 1 step = 6 frames (0.1s / (1/59.94)).
    df["step"] = ((df["frame"] - 300) / 6).round().astype(int)

    # Calculate Area
    df["area"] = df["width"] * df["height"]

    # Max-Pooling: Select box with largest area per player per step
    # Sort by area desc, then drop duplicates keeping first
    df.sort_values("area", ascending=False, inplace=True)
    df_pooled = df.drop_duplicates(
        subset=["game_play", "step", "nfl_player_id"], keep="first"
    ).copy()

    # Sort for shifting
    df_pooled.sort_values(["game_play", "nfl_player_id", "step"], inplace=True)

    feats_to_lag = Config.VISUAL_FEATURES

    grouped = df_pooled.groupby(["game_play", "nfl_player_id"])

    lagged_dfs = []
    base_cols = ["game_play", "nfl_player_id", "step"]
    lagged_dfs.append(df_pooled[base_cols])

    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        shifted = grouped[feats_to_lag].shift(-lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in feats_to_lag]
        lagged_dfs.append(shifted)

    df_wide = pd.concat(lagged_dfs, axis=1)
    df_wide = df_wide.fillna(0)

    return df_wide


def merge_and_create_pairs(metadata_df, track_df, vis_df):
    """
    Merges P1 and P2 data, handles Ground imputation, computes interaction features.
    """
    # Ensure IDs are correct types for merge
    metadata_df["nfl_player_id_1"] = pd.to_numeric(
        metadata_df["nfl_player_id_1"], errors="coerce"
    )
    # P2 can be 'G', so keep as object temporarily, but convert for merge
    metadata_df["nfl_player_id_2_str"] = metadata_df["nfl_player_id_2"].astype(str)

    # --- Merge Player 1 ---
    # Tracking
    merged = metadata_df.merge(
        track_df,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_drop"),
    )
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_drop")])

    # Rename P1 columns
    p1_kin_cols = [c for c in track_df.columns if "lag" in c]
    rename_map_p1 = {c: f"{c}_1" for c in p1_kin_cols}
    rename_map_p1.update({"position": "position_1", "team": "team_1"})
    merged.rename(columns=rename_map_p1, inplace=True)

    # Visuals P1
    merged = merged.merge(
        vis_df,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_drop"),
    )
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_drop")])
    p1_vis_cols = [c for c in vis_df.columns if "lag" in c]
    rename_map_vis_p1 = {c: f"{c}_1" for c in p1_vis_cols}
    merged.rename(columns=rename_map_vis_p1, inplace=True)

    # --- Merge Player 2 ---
    # Handle Ground vs Player
    is_ground = merged["nfl_player_id_2_str"] == "G"

    # Create temp numeric ID for merge (G becomes NaN)
    merged["nfl_player_id_2_num"] = pd.to_numeric(
        merged["nfl_player_id_2_str"], errors="coerce"
    )

    # Merge Tracking P2
    merged = merged.merge(
        track_df,
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_2"),
    )
    # Note: track_df columns don't have suffix in original, so they collide with P1 (already renamed)
    # or just appear. Because P1 cols were renamed to _1, the new cols are clean.
    # But we need to rename them to _2.
    p2_kin_cols = [c for c in track_df.columns if "lag" in c]
    rename_map_p2 = {c: f"{c}_2" for c in p2_kin_cols}
    rename_map_p2.update({"position": "position_2", "team": "team_2"})
    merged.rename(columns=rename_map_p2, inplace=True)

    # Merge Visuals P2
    merged = merged.merge(
        vis_df,
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_vis_2"),
    )
    p2_vis_cols = [c for c in vis_df.columns if "lag" in c]
    rename_map_vis_p2 = {c: f"{c}_2" for c in p2_vis_cols}
    merged.rename(columns=rename_map_vis_p2, inplace=True)

    # --- Hybrid Ground Imputation ---
    # If P2 is Ground:
    # Kinematics: Pos2 = Pos1, Speed2=0, Accel2=0, etc.
    # Visuals: 0

    lags = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    for lag in lags:
        # Kinematic Imputation
        # Position -> Same as P1
        merged.loc[is_ground, f"x_position_lag_{lag}_2"] = merged.loc[
            is_ground, f"x_position_lag_{lag}_1"
        ]
        merged.loc[is_ground, f"y_position_lag_{lag}_2"] = merged.loc[
            is_ground, f"y_position_lag_{lag}_1"
        ]

        # Velocity/Motion -> 0
        zero_cols = [
            "speed",
            "acceleration",
            "sa",
            "distance_traveled",
            "vx",
            "vy",
            "orientation_sin",
            "orientation_cos",
        ]
        for zc in zero_cols:
            merged.loc[is_ground, f"{zc}_lag_{lag}_2"] = 0

        # Visual Imputation -> 0
        for vc in Config.VISUAL_FEATURES:
            merged.loc[is_ground, f"{vc}_lag_{lag}_2"] = 0

    # Fill remaining NaNs (missing tracking for non-ground players) with 0
    merged.fillna(0, inplace=True)

    # --- Compute Interaction Features ---
    # We compute these for every lag

    interaction_cols_ordered = []

    for lag in lags:
        suffix_1 = f"_lag_{lag}_1"
        suffix_2 = f"_lag_{lag}_2"
        suffix_out = f"_lag_{lag}"

        x1 = merged[f"x_position{suffix_1}"]
        y1 = merged[f"y_position{suffix_1}"]
        x2 = merged[f"x_position{suffix_2}"]
        y2 = merged[f"y_position{suffix_2}"]

        vx1 = merged[f"vx{suffix_1}"]
        vy1 = merged[f"vy{suffix_1}"]
        vx2 = merged[f"vx{suffix_2}"]
        vy2 = merged[f"vy{suffix_2}"]

        # 1. Distance
        dist = calculate_euclidean_distance(x1, y1, x2, y2)
        merged[f"distance{suffix_out}"] = dist

        # 2. Log Distance
        merged[f"log_distance{suffix_out}"] = calculate_log_distance(dist)

        # 3. Relative Speed
        rel_vx = vx1 - vx2
        rel_vy = vy1 - vy2
        merged[f"relative_speed{suffix_out}"] = np.sqrt(rel_vx**2 + rel_vy**2)

        # 4. Closing Speed
        merged[f"closing_speed{suffix_out}"] = calculate_closing_speed(
            vx1, vy1, vx2, vy2, x1, y1, x2, y2
        )

        # 5. Diff X, Diff Y
        merged[f"diff_x{suffix_out}"] = x1 - x2
        merged[f"diff_y{suffix_out}"] = y1 - y2

        # Collect column names for ordering
        # Order: [P1_Feats, P2_Feats, Interaction_Feats] per lag

    # --- Final Column Assembly ---
    X_kin_cols = []
    X_vis_cols = []

    for lag in lags:
        # Kinematic Stream
        for feat in Config.KINEMATIC_FEATURES:
            X_kin_cols.append(f"{feat}_lag_{lag}_1")
        for feat in Config.KINEMATIC_FEATURES:
            X_kin_cols.append(f"{feat}_lag_{lag}_2")
        for feat in Config.INTERACTION_FEATURES:
            X_kin_cols.append(f"{feat}_lag_{lag}")

        # Visual Stream
        for feat in Config.VISUAL_FEATURES:
            X_vis_cols.append(f"{feat}_lag_{lag}_1")
        for feat in Config.VISUAL_FEATURES:
            X_vis_cols.append(f"{feat}_lag_{lag}_2")

    # Extract Arrays
    X_kin = merged[X_kin_cols].values.astype(np.float32)
    X_vis = merged[X_vis_cols].values.astype(np.float32)

    # Categorical Features
    # Position and Team need to be encoded
    # If ground, position_2 is 'Unknown' (mapped to len) or we can set to specific index.
    # We'll use the mapping.

    # Ensure columns exist (if tracking was missing, they might be 0/NaN)
    # We need to handle the case where position_1/2 are 0 (from fillna).
    # Convert 0 back to 'Unknown' or handle in get_pos_idx?
    # Actually, position columns are strings in tracking. fillna(0) made them 0 (int).
    # Let's cast to string.

    p1_pos = get_pos_idx(merged["position_1"].astype(str))
    p1_team = get_team_idx(merged["team_1"].astype(str))

    # For P2, if Ground, we want a distinct category? Or just 'Unknown'?
    # 'G' is not in POS_VOCAB, so it maps to Unknown. That works.
    p2_pos = get_pos_idx(merged["position_2"].astype(str))
    p2_team = get_team_idx(merged["team_2"].astype(str))

    X_cat = np.stack([p1_pos, p1_team, p2_pos, p2_team], axis=1).astype(np.int32)

    y = merged["contact"].values.astype(np.float32)
    ids = merged["contact_id"].values

    return X_kin, X_vis, X_cat, y, ids


def generate_features(split="train", load_cached_data=True, debug=False):
    """
    Main function to generate or load features.
    Args:
        split: 'train', 'validation', or 'test'
        load_cached_data: If True, attempts to load from disk.
        debug: If True, samples the data for quick testing.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # File paths
    f_kin = os.path.join(cache_dir, f"{split}_X_kin.npy")
    f_vis = os.path.join(cache_dir, f"{split}_X_vis.npy")
    f_cat = os.path.join(cache_dir, f"{split}_X_cat.npy")
    f_y = os.path.join(cache_dir, f"{split}_y.npy")
    f_ids = os.path.join(cache_dir, f"{split}_ids.npy")

    # Check cache
    if load_cached_data and os.path.exists(f_kin):
        print(f"Loading cached {split} features from {cache_dir}...")
        return (
            np.load(f_kin),
            np.load(f_vis),
            np.load(f_cat),
            np.load(f_y),
            np.load(f_ids, allow_pickle=True),
        )

    print(f"Generating {split} features from scratch...")

    # Load Metadata
    meta_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
    df_meta = pd.read_csv(meta_path)

    if debug:
        df_meta = df_meta.sample(5000, random_state=Config.SEED)

    unique_game_plays = df_meta["game_play"].unique()

    # Determine source files
    if split == "test":
        track_file = os.path.join(Config.INPUT_DIR, "test_player_tracking.csv")
        helm_file = os.path.join(Config.INPUT_DIR, "test_baseline_helmets.csv")
    else:
        track_file = os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
        helm_file = os.path.join(Config.INPUT_DIR, "train_baseline_helmets.csv")

    # Process Streams
    print("Processing Tracking Data...")
    df_track = process_tracking_data(track_file, unique_game_plays)

    print("Processing Visual Data...")
    df_vis = process_visual_data(helm_file, unique_game_plays)

    print("Merging and Computing Interactions...")
    X_kin, X_vis, X_cat, y, ids = merge_and_create_pairs(df_meta, df_track, df_vis)

    # Cache results
    print("Saving to cache...")
    np.save(f_kin, X_kin)
    np.save(f_vis, X_vis)
    np.save(f_cat, X_cat)
    np.save(f_y, y)
    np.save(f_ids, ids)

    return X_kin, X_vis, X_cat, y, ids
