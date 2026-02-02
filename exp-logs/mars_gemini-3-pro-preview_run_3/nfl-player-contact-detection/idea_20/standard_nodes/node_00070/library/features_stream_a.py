import pandas as pd
import numpy as np
import os
import sys
import library.config as C
import library.utils as U


def compute_iou_distance(df_main, df_helmets, view_name, prefix):
    """
    Merges helmet data for a specific view and computes IoU and Centroid Distance.
    """
    # Filter helmets by view
    helmets_view = df_helmets[df_helmets["view"] == view_name].copy()

    # Map frame to step (approximate: 300 is snap, 6 frames per 0.1s step)
    # frame 300 = step 0.
    # step = round((frame - 300) / 6)
    helmets_view["step"] = ((helmets_view["frame"] - 300) / 6).round().astype(int)

    # Select relevant columns and deduplicate (keep first detection if multiple)
    h_cols = ["game_play", "step", "nfl_player_id", "left", "width", "top", "height"]
    helmets_view = helmets_view[h_cols].drop_duplicates(
        subset=["game_play", "step", "nfl_player_id"]
    )

    # Prepare P1 and P2 merge
    # Merge P1
    df_merged = pd.merge(
        df_main,
        helmets_view.add_suffix("_p1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
        how="left",
    )

    # Merge P2
    df_merged = pd.merge(
        df_merged,
        helmets_view.add_suffix("_p2"),
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
        how="left",
    )

    # Calculate Box Coordinates
    # P1
    x1_p1 = df_merged["left_p1"]
    y1_p1 = df_merged["top_p1"]
    x2_p1 = x1_p1 + df_merged["width_p1"]
    y2_p1 = y1_p1 + df_merged["height_p1"]
    cx_p1 = x1_p1 + df_merged["width_p1"] / 2
    cy_p1 = y1_p1 + df_merged["height_p1"] / 2

    # P2
    x1_p2 = df_merged["left_p2"]
    y1_p2 = df_merged["top_p2"]
    x2_p2 = x1_p2 + df_merged["width_p2"]
    y2_p2 = y1_p2 + df_merged["height_p2"]
    cx_p2 = x1_p2 + df_merged["width_p2"] / 2
    cy_p2 = y1_p2 + df_merged["height_p2"] / 2

    # Intersection
    xi1 = np.maximum(x1_p1, x1_p2)
    yi1 = np.maximum(y1_p1, y1_p2)
    xi2 = np.minimum(x2_p1, x2_p2)
    yi2 = np.minimum(y2_p1, y2_p2)

    inter_width = np.maximum(0, xi2 - xi1)
    inter_height = np.maximum(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Union
    area_p1 = df_merged["width_p1"] * df_merged["height_p1"]
    area_p2 = df_merged["width_p2"] * df_merged["height_p2"]
    union = area_p1 + area_p2 - intersection

    # IoU
    # Avoid division by zero
    iou = np.where(union > 0, intersection / union, 0.0)

    # Centroid Distance (Pixel space)
    dist = np.sqrt((cx_p1 - cx_p2) ** 2 + (cy_p1 - cy_p2) ** 2)

    # Assign to dataframe
    # Fill NaNs (missing detections) with sentinel -999
    df_main[f"{prefix}_iou"] = np.nan_to_num(iou, nan=-999.0).astype(np.float32)
    df_main[f"{prefix}_dist"] = np.nan_to_num(dist, nan=-999.0).astype(np.float32)

    # If either player was missing, ensure sentinel is set (though nan_to_num handles NaNs from calc)
    # Explicitly check for missing source data
    missing_mask = df_merged["left_p1"].isna() | df_merged["left_p2"].isna()
    df_main.loc[missing_mask.values, f"{prefix}_iou"] = -999.0
    df_main.loc[missing_mask.values, f"{prefix}_dist"] = -999.0

    return df_main


def generate_stream_a_features(
    df_stream_a, df_helmets, mode="train", load_cached_data=True
):
    """
    Generates features for Stream A (Player-Player Interaction).

    Args:
        df_stream_a (pd.DataFrame): Merged tracking data for player pairs.
        df_helmets (pd.DataFrame): Helmet bounding box data.
        mode (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): Feature matrix.
            y (np.array): Target vector.
            ids (np.array): Contact IDs.
    """
    # 1. Caching Setup
    cache_dir = C.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_X = os.path.join(cache_dir, f"features_stream_a_{mode}_X.parquet")
    cache_path_y = os.path.join(cache_dir, f"features_stream_a_{mode}_y.npy")
    cache_path_ids = os.path.join(cache_dir, f"features_stream_a_{mode}_ids.npy")

    if load_cached_data:
        if (
            os.path.exists(cache_path_X)
            and os.path.exists(cache_path_y)
            and os.path.exists(cache_path_ids)
        ):
            print(f"Loading Stream A features from cache for {mode}...")
            X = pd.read_parquet(cache_path_X)
            y = np.load(cache_path_y)
            ids = np.load(cache_path_ids, allow_pickle=True)
            return X, y, ids
        else:
            print(f"Cache miss for Stream A ({mode}). Generating features...")

    print(f"Generating Stream A features for {len(df_stream_a)} rows...")

    # Working copy
    df = df_stream_a.copy().reset_index(drop=True)

    # 2. Kinematic Feature Engineering (Relational Scalars)
    # Euclidean Distance
    df["dist_p1_p2"] = np.sqrt(
        (df["x_position_p1"] - df["x_position_p2"]) ** 2
        + (df["y_position_p1"] - df["y_position_p2"]) ** 2
    ).astype(np.float32)

    # Speed Difference
    df["speed_diff"] = np.abs(df["speed_p1"] - df["speed_p2"]).astype(np.float32)

    # Cyclical Encoding (Orientation & Direction)
    # Convert degrees to radians
    for p in ["p1", "p2"]:
        for col in ["orientation", "direction"]:
            rad = np.deg2rad(df[f"{col}_{p}"].fillna(0))
            df[f"{col}_{p}_sin"] = np.sin(rad).astype(np.float32)
            df[f"{col}_{p}_cos"] = np.cos(rad).astype(np.float32)

    # Alignment Features (Cite 00027)
    # Pose-Motion Alignment: cos(orientation - direction)
    for p in ["p1", "p2"]:
        op = df[f"orientation_{p}"].fillna(0)
        dp = df[f"direction_{p}"].fillna(0)
        df[f"cos_orient_dir_{p}"] = np.cos(np.deg2rad(op - dp)).astype(np.float32)

    # Orientation Similarity: cos(orient_p1 - orient_p2)
    op1 = df["orientation_p1"].fillna(0)
    op2 = df["orientation_p2"].fillna(0)
    df["cos_sim_orient"] = np.cos(np.deg2rad(op1 - op2)).astype(np.float32)

    # 3. Visual Feature Engineering (Merge Helmets)
    print("Computing visual features (Sideline)...")
    df = compute_iou_distance(df, df_helmets, "Sideline", "sl")

    print("Computing visual features (Endzone)...")
    df = compute_iou_distance(df, df_helmets, "Endzone", "ez")

    # 4. Temporal Context (Lags & Derivatives)
    # Create Pair ID for grouping
    df["pair_id"] = (
        df["game_play"]
        + "_"
        + df["nfl_player_id_1"].astype(str)
        + "_"
        + df["nfl_player_id_2"].astype(str)
    )

    # Sort for shift operations
    df = df.sort_values(["game_play", "pair_id", "step"]).reset_index(drop=True)

    # Closure Rate (Derivative of Distance)
    # We calculate this before other lags
    # Groupby shift is safer than simple shift to avoid bleeding across plays
    print("Computing closure rate...")
    dist_lag1 = df.groupby(["game_play", "pair_id"])["dist_p1_p2"].shift(1)
    # Rate = (dist(t-1) - dist(t)) / 0.1s. Positive = Closing in.
    # We use simple difference as it's monotonic with rate.
    df["closure_rate"] = (dist_lag1 - df["dist_p1_p2"]).fillna(0).astype(np.float32)

    # Apply Sparse Lags to Base Features
    # Base features defined in Config
    base_features = C.STREAM_A_BASE_FEATURES
    lags = C.LAG_OFFSETS  # [0, 1, 2, 4, 8, 15]

    print("Applying temporal pyramids (lags)...")
    # We process lags in a loop.
    # Optimization: Groupby once, then apply shifts.
    grouper = df.groupby(["game_play", "pair_id"])

    for feature in base_features:
        if feature not in df.columns:
            # Should not happen if logic is correct, but safety check
            print(f"Warning: Base feature {feature} missing. Filling 0.")
            df[feature] = 0.0

        for lag in lags:
            if lag == 0:
                continue

            # Future Lag (t + k) -> shift(-k)
            col_name_future = f"{feature}_lag_{lag}"
            df[col_name_future] = (
                grouper[feature].shift(-lag).fillna(0).astype(np.float32)
            )

            # Past Lag (t - k) -> shift(k)
            col_name_past = f"{feature}_lag_minus_{lag}"
            df[col_name_past] = grouper[feature].shift(lag).fillna(0).astype(np.float32)

    # 5. Final Selection & Cleaning
    # Ensure all expected columns exist
    expected_cols = C.STREAM_A_COLS

    # Check for missing columns and fill 0
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} expected columns. Filling with 0.")
        for c in missing_cols:
            df[c] = 0.0

    # Select X
    X = df[expected_cols].copy()

    # Select y (target) - handle test mode where contact might be placeholder
    if "contact" in df.columns:
        y = df["contact"].values.astype(np.int8)
    else:
        y = np.zeros(len(df), dtype=np.int8)

    # Select IDs
    # Construct contact_id if not present, or use existing
    if "contact_id" in df.columns:
        ids = df["contact_id"].values
    else:
        # Reconstruct: game_play_step_p1_p2
        ids = (
            df["game_play"]
            + "_"
            + df["step"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        ).values

    # Memory Optimization
    X = U.reduce_mem_usage(X, verbose=False)

    # 6. Save to Cache
    print(f"Saving Stream A features to {cache_dir}...")
    X.to_parquet(cache_path_X, index=False)
    np.save(cache_path_y, y)
    np.save(cache_path_ids, ids)

    return X, y, ids
