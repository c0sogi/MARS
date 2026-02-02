import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import reduce_mem_usage


def process_tracking(df_tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Applies 'Entity-First' processing to tracking data:
    1. Sorts data.
    2. Generates windowed features (lags/futures).
    3. Drops non-invariant categorical columns.
    """
    # Sort for windowing
    df_tracking = df_tracking.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    # Columns to window
    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Identify boundaries for shifting
    # We create a grouping ID. If game_play or nfl_player_id changes, it's a new group.
    # However, since we sorted, we can just use groupby.
    grouper = df_tracking.groupby(["game_play", "nfl_player_id"])

    # Generate Lags and Futures
    # Window: t-WINDOW_LAG to t+WINDOW_FUTURE
    # We flatten these into wide columns: col_t-5, col_t-4, ... col_t+5

    window_features = []

    # We iterate from -LAG to +FUTURE
    # e.g. -5 to +5
    start_lag = -Config.WINDOW_LAG
    end_lag = Config.WINDOW_FUTURE

    for shift_step in range(start_lag, end_lag + 1):
        suffix = f"_t{shift_step:+d}" if shift_step != 0 else ""

        # Shift the features
        # shift(negative) gets future, shift(positive) gets past in pandas
        # We want t-5 (past) to be available at t.
        # To get value at t-5, we shift by +5.
        # To get value at t+5, we shift by -5.
        # So pandas shift amount = -1 * shift_step

        shifted = grouper[feature_cols].shift(-shift_step)
        shifted.columns = [f"{c}{suffix}" for c in feature_cols]
        window_features.append(shifted)

    df_windowed = pd.concat(window_features, axis=1)

    # Concatenate back to keys
    # We only keep keys and the windowed features
    keep_cols = ["game_play", "nfl_player_id", "step"]
    df_processed = pd.concat([df_tracking[keep_cols], df_windowed], axis=1)

    # Handle NaNs created by shifting (edges of plays)
    # We fill with the nearest valid observation (ffill/bfill) or 0?
    # Usually ffill/bfill within group is safer for trajectories, but simple 0 fill
    # for missing steps is robust enough if normalized later.
    # Given the strict physical constraints, 0 fill for missing edges is acceptable
    # as long as we don't fill across players.
    # Actually, the shift introduced NaNs. Let's fill with 0 for now,
    # the scaler will handle distribution.
    df_processed = df_processed.fillna(0)

    return df_processed


def process_visuals(df_helmets: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Max-Pooling Selection Strategy to helmet boxes.
    Selects the view with the largest box area for each player/frame.
    """
    # Calculate area for max pooling
    df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

    # Sort by area descending so the first record is the largest
    df_helmets = df_helmets.sort_values(
        ["game_play", "frame", "nfl_player_id", "area"],
        ascending=[True, True, True, False],
    )

    # Drop duplicates to keep only the largest area per player per frame
    # We keep 'left', 'top', 'width', 'height'
    cols_to_keep = [
        "game_play",
        "frame",
        "nfl_player_id",
        "left",
        "top",
        "width",
        "height",
    ]
    df_pooled = df_helmets.drop_duplicates(
        subset=["game_play", "frame", "nfl_player_id"], keep="first"
    )[cols_to_keep]

    return df_pooled


def impute_ground(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles ground contact logic:
    - If P2 is Ground ('G'), P2 pos = P1 pos, P2 velocity/accel = 0.
    - P2 visual features set to 0.
    """
    # Identify Ground rows
    is_ground = df["nfl_player_id_2"] == "G"

    if not is_ground.any():
        return df

    # --- Kinematic Imputation ---
    # List of base features that were windowed
    base_feats = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Reconstruct all windowed column names for P2
    p2_cols = []
    p1_cols = []

    start_lag = -Config.WINDOW_LAG
    end_lag = Config.WINDOW_FUTURE

    for shift_step in range(start_lag, end_lag + 1):
        suffix = f"_t{shift_step:+d}" if shift_step != 0 else ""
        for feat in base_feats:
            p2_cols.append(f"{feat}{suffix}_2")
            p1_cols.append(f"{feat}{suffix}_1")

    # We can't vectorized assign efficiently with mixed types if we iterate.
    # Instead, we use numpy masking.

    # 1. Position: Set P2 position to P1 position for Ground
    # We need to match the specific columns.
    # x_position_tX_2 should equal x_position_tX_1
    # y_position_tX_2 should equal y_position_tX_1

    # 2. Dynamics: Set Speed, Accel, SA to 0
    # 3. Angles: Set to 0 (neutral)

    for shift_step in range(start_lag, end_lag + 1):
        suffix = f"_t{shift_step:+d}" if shift_step != 0 else ""

        # Position Copy
        df.loc[is_ground, f"x_position{suffix}_2"] = df.loc[
            is_ground, f"x_position{suffix}_1"
        ]
        df.loc[is_ground, f"y_position{suffix}_2"] = df.loc[
            is_ground, f"y_position{suffix}_1"
        ]

        # Zero out dynamics and angles for Ground
        zero_feats = ["speed", "direction", "orientation", "acceleration", "sa"]
        for zf in zero_feats:
            df.loc[is_ground, f"{zf}{suffix}_2"] = 0.0

    # --- Visual Imputation ---
    # Set P2 visual features to 0 for Ground
    vis_cols = ["left", "top", "width", "height"]
    for vc in vis_cols:
        if f"{vc}_2" in df.columns:
            df.loc[is_ground, f"{vc}_2"] = 0.0

    return df


def calculate_iou(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Intersection over Union (IoU) for P1 and P2 bounding boxes.
    """
    # P1 boxes
    x1_p1 = df["left_1"]
    y1_p1 = df["top_1"]
    x2_p1 = df["left_1"] + df["width_1"]
    y2_p1 = df["top_1"] + df["height_1"]
    area_p1 = df["width_1"] * df["height_1"]

    # P2 boxes
    x1_p2 = df["left_2"]
    y1_p2 = df["top_2"]
    x2_p2 = df["left_2"] + df["width_2"]
    y2_p2 = df["top_2"] + df["height_2"]
    area_p2 = df["width_2"] * df["height_2"]

    # Intersection
    x1_i = np.maximum(x1_p1, x1_p2)
    y1_i = np.maximum(y1_p1, y1_p2)
    x2_i = np.minimum(x2_p1, x2_p2)
    y2_i = np.minimum(y2_p1, y2_p2)

    w_i = np.maximum(0, x2_i - x1_i)
    h_i = np.maximum(0, y2_i - y1_i)
    intersection = w_i * h_i

    # Union
    union = area_p1 + area_p2 - intersection

    # Avoid division by zero
    iou = intersection / (union + 1e-6)

    # If P2 is Ground (area_p2 is 0), IoU should be 0
    # The formula works naturally since intersection will be 0.

    return iou


def prepare_features(
    train_mode: bool = True, load_cached_data: bool = True, debug: bool = False
):
    """
    Main pipeline to prepare features.

    Args:
        train_mode (bool): If True, processes training data. Else test data.
        load_cached_data (bool): If True, attempts to load from cache.
        debug (bool): If True, samples the data for quick debugging.
    """
    # Define paths
    if train_mode:
        meta_path = Config.TRAIN_META_PATH
        track_path = Config.TRAIN_TRACKING_PATH
        helmets_path = Config.TRAIN_HELMETS_PATH
        cache_path = Config.CACHE_TRAIN_FEATURES
    else:
        meta_path = Config.TEST_META_PATH
        track_path = Config.TEST_TRACKING_PATH
        helmets_path = Config.TEST_HELMETS_PATH
        cache_path = Config.CACHE_TEST_FEATURES

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        df_final = pd.read_parquet(cache_path)

        # Separate X, y, ids
        feature_cols = [
            c
            for c in df_final.columns
            if c
            not in [
                "contact",
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
            ]
        ]
        X = df_final[feature_cols].values.astype(np.float32)
        ids = df_final["contact_id"].values

        if "contact" in df_final.columns:
            y = df_final["contact"].values.astype(np.float32)
            return X, y, ids
        else:
            return X, None, ids

    # 2. Load Raw Data
    print("Loading raw data...")
    df_meta = pd.read_csv(meta_path)
    df_tracking = pd.read_csv(track_path)
    df_helmets = pd.read_csv(helmets_path)

    if debug:
        print("Debug mode: Sampling data...")
        df_meta = df_meta.sample(5000, random_state=Config.SEED).reset_index(drop=True)
        # Filter tracking/helmets to match sampled plays
        valid_plays = df_meta["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(valid_plays)].copy()
        df_helmets = df_helmets[df_helmets["game_play"].isin(valid_plays)].copy()

    # 3. Process Streams
    print("Processing tracking data (Entity-First)...")
    df_track_proc = process_tracking(df_tracking)

    print("Processing visual data (Max-Pooling)...")
    df_vis_proc = process_visuals(df_helmets)

    # 4. Merge Data
    print("Merging data...")
    # Prepare metadata for merge
    # Ensure nfl_player_id_1 is numeric
    df_meta["nfl_player_id_1"] = (
        pd.to_numeric(df_meta["nfl_player_id_1"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Handle P2 (can be 'G')
    # Create a numeric column for merge, 'G' becomes NaN -> -1
    df_meta["nfl_player_id_2_num"] = (
        pd.to_numeric(df_meta["nfl_player_id_2"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Merge P1 Tracking
    df_merged = df_meta.merge(
        df_track_proc.add_suffix("_1"),
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play_1", "nfl_player_id_1", "step_1"],
        how="left",
    )

    # Merge P2 Tracking
    df_merged = df_merged.merge(
        df_track_proc.add_suffix("_2"),
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play_2", "nfl_player_id_2", "step_2"],
        how="left",
    )

    # Map Step to Frame for Visual Merge
    # Frame approx = 300 + step * 5.994
    df_merged["frame_approx"] = (300 + df_merged["step"] * 5.994).round().astype(int)

    # Merge P1 Visuals
    df_merged = df_merged.merge(
        df_vis_proc.add_suffix("_1"),
        left_on=["game_play", "frame_approx", "nfl_player_id_1"],
        right_on=["game_play_1", "frame_1", "nfl_player_id_1"],
        how="left",
    )

    # Merge P2 Visuals
    df_merged = df_merged.merge(
        df_vis_proc.add_suffix("_2"),
        left_on=["game_play", "frame_approx", "nfl_player_id_2_num"],
        right_on=["game_play_2", "frame_2", "nfl_player_id_2"],
        how="left",
    )

    # 5. Ground Imputation
    print("Imputing ground interactions...")
    df_merged = impute_ground(df_merged)

    # 6. Feature Engineering
    print("Generating derived features...")

    # Fill NaNs before calculation (tracking might be missing)
    # We fill with 0, as scaler handles it, and 0 is physically neutral for velocity/accel
    # For positions, 0 is bad, but relative features handle it.
    # Actually, if tracking is missing, relative dist is invalid.
    # But we must produce a prediction.
    num_cols = [
        c
        for c in df_merged.columns
        if "position" in c
        or "speed" in c
        or "accel" in c
        or "orient" in c
        or "direct" in c
        or "sa" in c
        or "left" in c
        or "top" in c
        or "width" in c
        or "height" in c
    ]
    df_merged[num_cols] = df_merged[num_cols].fillna(0)

    # Calculate IoU
    df_merged["visual_iou"] = calculate_iou(df_merged)

    # Relative Kinematics (Windowed)
    start_lag = -Config.WINDOW_LAG
    end_lag = Config.WINDOW_FUTURE

    for shift_step in range(start_lag, end_lag + 1):
        suffix = f"_t{shift_step:+d}" if shift_step != 0 else ""

        # Distance
        dx = df_merged[f"x_position{suffix}_1"] - df_merged[f"x_position{suffix}_2"]
        dy = df_merged[f"y_position{suffix}_1"] - df_merged[f"y_position{suffix}_2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Log1p Distance (Resolution Enhancement)
        df_merged[f"log_dist{suffix}"] = np.log1p(dist)

        # Speed Diff
        df_merged[f"speed_diff{suffix}"] = (
            df_merged[f"speed{suffix}_1"] - df_merged[f"speed{suffix}_2"]
        )

        # Angular Diff (Shortest Arc)
        # Orientation
        o1 = df_merged[f"orientation{suffix}_1"]
        o2 = df_merged[f"orientation{suffix}_2"]
        diff_o = np.abs(o1 - o2)
        df_merged[f"orient_diff{suffix}"] = np.minimum(diff_o, 360 - diff_o)

        # Direction
        d1 = df_merged[f"direction{suffix}_1"]
        d2 = df_merged[f"direction{suffix}_2"]
        diff_d = np.abs(d1 - d2)
        df_merged[f"direct_diff{suffix}"] = np.minimum(diff_d, 360 - diff_d)

    # 7. Clamping (Physical Constraints)
    # Clamp all numeric features to range
    print("Applying physical clamping...")
    feature_cols = [
        c
        for c in df_merged.columns
        if "_t" in c
        or "visual_" in c
        or "width" in c
        or "height" in c
        or "top" in c
        or "left" in c
    ]
    # Exclude ID columns
    feature_cols = [
        c for c in feature_cols if "game_play" not in c and "nfl_player_id" not in c
    ]

    # Add raw kinematic columns if they aren't in feature_cols yet (they have _t suffix so they are)

    # Clamp
    df_merged[feature_cols] = df_merged[feature_cols].clip(
        Config.CLAMP_MIN, Config.CLAMP_MAX
    )

    # 8. Normalization
    print("Normalizing features...")
    X = df_merged[feature_cols].values.astype(np.float32)

    if train_mode:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, Config.SCALER_SAVE_PATH)
    else:
        # Load scaler
        if os.path.exists(Config.SCALER_SAVE_PATH):
            scaler = joblib.load(Config.SCALER_SAVE_PATH)
            X = scaler.transform(X)
        else:
            print(
                "Warning: Scaler not found. Using raw data (this is bad for inference)."
            )

    # 9. Save to Cache
    print(f"Saving features to {cache_path}...")
    # Construct DataFrame for saving
    df_save = pd.DataFrame(X, columns=feature_cols)
    df_save["contact_id"] = df_merged["contact_id"]

    # Keep essential IDs for debugging/splitting
    df_save["game_play"] = df_merged["game_play"]
    df_save["step"] = df_merged["step"]
    df_save["nfl_player_id_1"] = df_merged["nfl_player_id_1"]
    df_save["nfl_player_id_2"] = df_merged["nfl_player_id_2"]

    if "contact" in df_merged.columns:
        df_save["contact"] = df_merged["contact"]
        y = df_merged["contact"].values.astype(np.float32)
    else:
        y = None

    ids = df_merged["contact_id"].values

    # Optimize memory before save
    df_save = reduce_mem_usage(df_save)
    df_save.to_parquet(cache_path, index=False)

    return X, y, ids
