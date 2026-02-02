import os
import numpy as np
import pandas as pd
from library.config import Config

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)


def angular_diff(a, b):
    """Computes shortest arc difference between two angles in degrees."""
    diff = np.abs(a - b)
    return np.minimum(diff, 360 - diff)


def process_tracking_data(df_tracking):
    """
    Applies Entity-First strategy: Generates lag features on the tracking dataframe
    BEFORE merging with labels to avoid O(N^2) complexity.
    """
    # Sort to ensure correct shifting
    df_tracking = df_tracking.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    # Base features to lag
    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "distance",
        "sa",
    ]

    # We will construct a list of dataframes to concat (more efficient than repeated assignment)
    dfs_to_concat = []

    # Keep the key columns
    keys = df_tracking[["game_play", "nfl_player_id", "step"]]
    dfs_to_concat.append(keys)

    # Group object for shifting
    # Note: We assume the data is dense per player per play.
    # If steps are missing, shift() aligns by index position, not value.
    # Given the dataset description, steps are 0.1s increments.
    # We assume continuity for efficiency, or we would need reindexing.
    grp = df_tracking.groupby(["game_play", "nfl_player_id"])

    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"

        # Select features and shift
        # lag < 0: Future (shift negative values)
        # lag > 0: Past (shift positive values)
        shifted = grp[feature_cols].shift(lag)

        # Rename columns
        shifted.columns = [f"{c}{suffix}" for c in feature_cols]

        # Fill edges with 0 (or could use ffill/bfill, but 0 is safer for "missing info")
        shifted = shifted.fillna(0)

        dfs_to_concat.append(shifted)

    # Concatenate all features horizontally
    df_wide = pd.concat(dfs_to_concat, axis=1)

    return df_wide


def process_visual_data(df_helmets):
    """
    Processes helmet bounding boxes:
    1. Maps frame to step.
    2. Calculates Area.
    3. Performs Max-Pooling (select best view per player/step).
    4. Normalizes coordinates.
    """
    # Map frame to step: step 0 is frame 300, 10Hz labels vs 59.94Hz video
    # step = round((frame - 300) / 6)
    df_helmets["step"] = ((df_helmets["frame"] - 300) / 6).round().astype(int)

    # Calculate Area
    df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

    # Max Pooling: Sort by area desc, then drop duplicates on keys
    # This keeps the row with the largest area for each (game_play, player, step)
    df_best = df_helmets.sort_values("area", ascending=False).drop_duplicates(
        subset=["game_play", "nfl_player_id", "step"]
    )

    # Select and Normalize features
    # Image dims: 1280x720
    vis_cols = ["left", "width", "top", "height"]
    df_out = df_best[["game_play", "nfl_player_id", "step"] + vis_cols].copy()

    df_out["left"] = df_out["left"] / 1280.0
    df_out["width"] = df_out["width"] / 1280.0
    df_out["top"] = df_out["top"] / 720.0
    df_out["height"] = df_out["height"] / 720.0

    return df_out


def merge_and_impute(df_labels, df_tracking_wide, df_visuals):
    """
    Merges labels with processed tracking and visual data.
    Handles Ground imputation and Relative Physics calculation.
    """
    # 1. Merge P1 Tracking
    df_merged = df_labels.merge(
        df_tracking_wide,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns
    # Identify lag columns
    lag_cols = [c for c in df_tracking_wide.columns if "_lag_" in c]
    rename_map_p1 = {c: f"p1_{c}" for c in lag_cols}
    df_merged = df_merged.rename(columns=rename_map_p1)

    # 2. Merge P2 Tracking
    # Handle Ground ID 'G' -> NaN for merge
    df_merged["nfl_player_id_2_num"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    df_merged = df_merged.merge(
        df_tracking_wide,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_p2"),
    ).drop(columns=["nfl_player_id"])

    # Rename P2 columns
    rename_map_p2 = {c: f"p2_{c}" for c in lag_cols}
    df_merged = df_merged.rename(columns=rename_map_p2)

    # 3. Merge P1 Visuals
    # Visuals are only used for P1 (the "subject" of the contact check usually, or we just use P1 visuals as proxy)
    # The idea description says "Visual Stream... flattened wide feature vector...".
    # Usually we want visuals for the players involved.
    # The Config suggests we just merge visuals. Let's merge for P1 as the primary visual anchor.
    # If we needed P2 visuals, we would merge them too, but typically helmet data is sparse/noisy.
    # We will stick to P1 visuals as per common baseline strategies unless specified otherwise.

    vis_cols = ["left", "width", "top", "height"]
    df_merged = df_merged.merge(
        df_visuals,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename Visuals
    rename_map_vis = {c: f"v_{c}" for c in vis_cols}
    df_merged = df_merged.rename(columns=rename_map_vis)

    # Fill missing visuals with 0
    for c in rename_map_vis.values():
        df_merged[c] = df_merged[c].fillna(0.0)

    # 4. Ground Imputation & Relative Physics
    is_ground = df_merged["nfl_player_id_2"] == "G"

    base_feats = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "distance",
        "sa",
    ]

    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"

        # Columns for this lag
        p1_x = f"p1_x_position{suffix}"
        p1_y = f"p1_y_position{suffix}"
        p2_x = f"p2_x_position{suffix}"
        p2_y = f"p2_y_position{suffix}"

        # Impute Ground P2
        # Position: P2 = P1 (Distance becomes 0)
        df_merged.loc[is_ground, p2_x] = df_merged.loc[is_ground, p1_x]
        df_merged.loc[is_ground, p2_y] = df_merged.loc[is_ground, p1_y]

        # Dynamics: P2 = 0
        for feat in [
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "distance",
            "sa",
        ]:
            p2_col = f"p2_{feat}{suffix}"
            df_merged.loc[is_ground, p2_col] = 0.0

        # Fill remaining NaNs (missing tracking for non-ground) with 0
        # We do this for all columns in this lag group
        for feat in base_feats:
            df_merged[f"p1_{feat}{suffix}"] = df_merged[f"p1_{feat}{suffix}"].fillna(0)
            df_merged[f"p2_{feat}{suffix}"] = df_merged[f"p2_{feat}{suffix}"].fillna(0)

        # Calculate Relative Features
        # Distance
        dx = df_merged[p1_x] - df_merged[p2_x]
        dy = df_merged[p1_y] - df_merged[p2_y]
        dist = np.sqrt(dx**2 + dy**2)

        df_merged[f"log_dist{suffix}"] = np.log1p(dist)

        # Relative Speed
        s1 = df_merged[f"p1_speed{suffix}"]
        s2 = df_merged[f"p2_speed{suffix}"]
        df_merged[f"rel_speed{suffix}"] = np.abs(s1 - s2)

        # Relative Angle (Shortest Arc)
        d1 = df_merged[f"p1_direction{suffix}"]
        d2 = df_merged[f"p2_direction{suffix}"]
        df_merged[f"rel_angle{suffix}"] = angular_diff(d1, d2)

    # Cleanup
    drop_cols = [
        "nfl_player_id_2_num",
        "datetime",
        "path_endzone",
        "path_sideline",
        "path_all29",
    ]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    return df_merged


def generate_dataset(mode="train", load_cached_data=True):
    """
    Main entry point for feature engineering.
    mode: 'train' (returns train+val combined), 'test'
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_features.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating {mode} features from scratch...")

    # 1. Load Metadata
    if mode == "train":
        df_meta = pd.read_csv(Config.METADATA_TRAIN)
        df_val = pd.read_csv(Config.METADATA_VAL)
        df_labels = pd.concat([df_meta, df_val], ignore_index=True)
        tracking_path = Config.TRAIN_TRACKING
        helmets_path = Config.TRAIN_HELMETS
    else:
        df_labels = pd.read_csv(Config.METADATA_TEST)
        tracking_path = Config.TEST_TRACKING
        helmets_path = Config.TEST_HELMETS

    # 2. Load Raw Data
    # Filter tracking/helmets to relevant game_plays to save memory
    relevant_gps = df_labels["game_play"].unique()

    print("Loading and filtering tracking data...")
    df_tracking = pd.read_csv(tracking_path, usecols=Config.TRACKING_COLS)
    df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_gps)].copy()

    print("Loading and filtering helmet data...")
    df_helmets = pd.read_csv(helmets_path)
    df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_gps)].copy()

    # 3. Process Streams
    print("Processing tracking data (Entity-First)...")
    df_tracking_wide = process_tracking_data(df_tracking)

    print("Processing visual data (Max-Pooling)...")
    df_visuals = process_visual_data(df_helmets)

    # 4. Merge and Impute
    print("Merging and creating relative features...")
    df_final = merge_and_impute(df_labels, df_tracking_wide, df_visuals)

    # 5. Cache
    print(f"Saving to {cache_file}...")
    df_final.to_parquet(cache_file)

    return df_final
