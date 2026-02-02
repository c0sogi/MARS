import os
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library import config

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

RAW_TRACKING_COLS = [
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "orientation",
    "direction",
    "sa",
]

RAW_HELMET_COLS = ["left", "width", "top", "height"]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _create_lags(df, group_cols, feature_cols, window_size=5):
    """
    Generates lag features for the specified columns.
    Assumes df is already sorted by [group_cols, 'step'].
    """
    df_out = df[group_cols + ["step"]].copy()

    # We need to ensure the data is strictly sorted and grouped
    # Using shift within groups is slow, so we use global shift with masking
    # But first, let's just use the index if possible or simple shifts if data is dense
    # Given the data structure, grouping by game_play and nfl_player_id is necessary.

    # To optimize, we can rely on the fact that steps are sequential integers.
    # However, missing steps might exist. Let's use a robust method: pivot or reindexing?
    # No, reindexing is memory heavy.
    # Let's use groupby shift. With optimized pandas it's decent.

    grouped = df.groupby(group_cols, sort=False)

    for lag in range(-window_size, window_size + 1):
        suffix = f"_lag_{lag}"
        for col in feature_cols:
            col_name = f"{col}{suffix}"
            # negative lag = future (shift -k), positive lag = past (shift k) in some conventions
            # Here: lag -5 means t-5 (past), lag +5 means t+5 (future)
            # shift(1) moves t to t+1 (getting previous value).
            # So shift(lag) gets value from 'lag' steps ago.
            df_out[col_name] = grouped[col].shift(lag)

    return df_out


def _process_tracking_entity_first(tracking_path, game_plays=None):
    """
    Loads tracking data, filters, sorts, and generates windowed features.
    """
    print(f"Processing tracking data from {tracking_path}...")
    df = pd.read_csv(tracking_path)

    if game_plays is not None:
        df = df[df["game_play"].isin(game_plays)].copy()

    # Ensure IDs are strings for consistent merging
    df["nfl_player_id"] = df["nfl_player_id"].astype(str)

    # Sort for windowing
    df = df.sort_values(["game_play", "nfl_player_id", "step"])

    # Generate lags
    # We want features at t-5 ... t ... t+5
    # Using a helper that does groupby shift
    df_lags = _create_lags(
        df,
        group_cols=["game_play", "nfl_player_id"],
        feature_cols=RAW_TRACKING_COLS,
        window_size=config.WINDOW_SIZE,
    )

    # Fill NaNs in tracking (e.g. edges of play) with nearest or linear?
    # Or just 0. Forward fill/Back fill is safer for time series.
    # Groupby fill is slow. Let's fill with 0 for now as simple imputation,
    # or leave as NaN and handle after merge (but merge might drop rows if inner).
    # We will use left join on labels, so rows won't be dropped.
    # We'll fill NaNs at the end.

    return df_lags


def _process_helmets_max_pooling(helmets_path, game_plays=None):
    """
    Loads helmet data, applies max pooling (best view), and generates windowed features.
    """
    print(f"Processing helmet data from {helmets_path}...")
    df = pd.read_csv(helmets_path)

    if game_plays is not None:
        df = df[df["game_play"].isin(game_plays)].copy()

    # Convert frame to step
    # Snap is at frame 300 (5s). Step is 10Hz. 5.994 frames per step.
    if "step" not in df.columns and "frame" in df.columns:
        df["step"] = ((df["frame"] - 300) / 5.994).round().astype(int)

    df["nfl_player_id"] = df["nfl_player_id"].astype(str)

    # Calculate Area
    df["area"] = df["width"] * df["height"]

    # Max Pooling: Select row with max area per player per step
    # Sort by area descending
    df = df.sort_values(
        ["game_play", "nfl_player_id", "step", "area"],
        ascending=[True, True, True, False],
    )

    # Drop duplicates keeping first (max area)
    df_pooled = df.drop_duplicates(
        subset=["game_play", "nfl_player_id", "step"], keep="first"
    ).copy()

    # Generate lags
    df_lags = _create_lags(
        df_pooled,
        group_cols=["game_play", "nfl_player_id"],
        feature_cols=RAW_HELMET_COLS + ["area"],
        window_size=config.WINDOW_SIZE,
    )

    return df_lags


def _compute_derived_features(df, suffix_p1, suffix_p2, out_suffix):
    """
    Computes relative kinematic and visual features for a specific lag.
    In-place modification of df.
    """
    # --- Kinematics ---
    # Coordinates
    x1 = df[f"x_position{suffix_p1}"]
    y1 = df[f"y_position{suffix_p1}"]
    x2 = df[f"x_position{suffix_p2}"]
    y2 = df[f"y_position{suffix_p2}"]

    # Ground Imputation Logic (Vectorized)
    # If is_ground is 1, we force P2 pos = P1 pos, P2 vel = 0
    # Note: 'is_ground' is a column in df
    is_g = df["is_ground"] == 1

    # Apply imputation to P2 raw features in the dataframe (if needed for debugging)
    # But mainly we need correct relative features.
    # Let's calculate diffs assuming values are correct, then override for Ground.

    dx = x1 - x2
    dy = y1 - y2

    # Override for Ground: dx=0, dy=0
    dx = np.where(is_g, 0.0, dx)
    dy = np.where(is_g, 0.0, dy)

    dist = np.sqrt(dx**2 + dy**2)
    df[f"distance{out_suffix}"] = dist
    df[f"log_distance{out_suffix}"] = np.log1p(dist)

    # Speed / Velocity
    s1 = df[f"speed{suffix_p1}"]
    s2 = df[f"speed{suffix_p2}"]
    # Impute P2 speed = 0 for Ground
    s2 = np.where(is_g, 0.0, s2)

    df[f"relative_speed{out_suffix}"] = s1 - s2  # Simple magnitude diff

    # Closing Speed
    # Need velocity vectors.
    # tracking: orientation (0-360, 0=y-axis, cw?), direction (motion angle)
    # NFL data: direction is 0..360, 0 = Y axis, clockwise.
    # Convert to radians.
    # dir=0 -> (0, 1), dir=90 -> (1, 0)
    def get_vec(speed, direction):
        rad = np.radians(direction)
        # 0 is North (Y+), 90 is East (X+)
        # sin(theta) gives X, cos(theta) gives Y
        vx = speed * np.sin(rad)
        vy = speed * np.cos(rad)
        return vx, vy

    vx1, vy1 = get_vec(s1, df[f"direction{suffix_p1}"])
    vx2, vy2 = get_vec(s2, df[f"direction{suffix_p2}"])

    # Ground velocity is 0
    vx2 = np.where(is_g, 0.0, vx2)
    vy2 = np.where(is_g, 0.0, vy2)

    # Relative velocity vector (P1 - P2)
    rvx = vx1 - vx2
    rvy = vy1 - vy2

    # Project onto position vector (P2 -> P1) ? Or P1 -> P2?
    # Closing speed: positive if getting closer.
    # Vector P1->P2 is (-dx, -dy).
    # Relative vel V1 - V2.
    # Dot product.
    # If P1 moving towards P2, V1 is aligned with P1->P2.
    # Let's use standard definition: - ( (v1-v2) dot (p1-p2) ) / |p1-p2|
    # p1-p2 is (dx, dy).
    dot_prod = rvx * dx + rvy * dy
    # Avoid div by zero
    safe_dist = np.where(dist < 1e-6, 1e-6, dist)
    closing = -(dot_prod / safe_dist)

    # If dist is 0 (Ground), closing speed is just speed_1 (impact speed)
    closing = np.where(dist < 1e-6, s1, closing)
    df[f"closing_speed{out_suffix}"] = closing

    # Orientation/Direction diffs
    o1 = df[f"orientation{suffix_p1}"]
    o2 = df[f"orientation{suffix_p2}"]
    # Handle Ground orientation? Undefined. Set to 0 or P1's?
    # Let's set diff to 0.
    o2 = np.where(is_g, o1, o2)
    # Cite solution_lesson_node_00076: Enforce numerical continuity for periodic variables
    diff_o = (o1 - o2) % 360
    df[f"relative_orientation{out_suffix}"] = np.minimum(diff_o, 360 - diff_o)

    d1 = df[f"direction{suffix_p1}"]
    d2 = df[f"direction{suffix_p2}"]
    d2 = np.where(is_g, d1, d2)
    diff_d = (d1 - d2) % 360
    df[f"relative_direction{out_suffix}"] = np.minimum(diff_d, 360 - diff_d)

    # Pass through is_ground for the lag (it's constant across lags but needed in feature list)
    df[f"is_ground{out_suffix}"] = is_g.astype(float)

    # Pass through raw P1/P2 features required by config
    # P1
    for col in RAW_TRACKING_COLS:
        df[f"{col}_1{out_suffix}"] = df[f"{col}{suffix_p1}"]
    # P2 (Imputed)
    for col in RAW_TRACKING_COLS:
        raw_val = df[f"{col}{suffix_p2}"]
        if col in ["x_position", "y_position"]:
            # P2 pos = P1 pos if Ground
            raw_val = np.where(is_g, df[f"{col}{suffix_p1}"], raw_val)
        else:
            # Speed, Accel, etc = 0 if Ground
            raw_val = np.where(is_g, 0.0, raw_val)
        df[f"{col}_2{out_suffix}"] = raw_val

    # --- Visuals ---
    # Box IoU
    # We have left, width, top, height
    l1 = df[f"left{suffix_p1}"]
    w1 = df[f"width{suffix_p1}"]
    t1 = df[f"top{suffix_p1}"]
    h1 = df[f"height{suffix_p1}"]
    r1 = l1 + w1
    b1 = t1 + h1

    l2 = df[f"left{suffix_p2}"]
    w2 = df[f"width{suffix_p2}"]
    t2 = df[f"top{suffix_p2}"]
    h2 = df[f"height{suffix_p2}"]
    r2 = l2 + w2
    b2 = t2 + h2

    # Intersection
    xi1 = np.maximum(l1, l2)
    yi1 = np.maximum(t1, t2)
    xi2 = np.minimum(r1, r2)
    yi2 = np.minimum(b1, b2)
    inter_area = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / np.where(union_area < 1e-6, 1e-6, union_area)

    # Ground Logic for Visuals
    # If Ground, IoU = 0
    iou = np.where(is_g, 0.0, iou)
    df[f"view_iou{out_suffix}"] = iou

    # Box Distance (centers)
    cx1 = l1 + w1 / 2
    cy1 = t1 + h1 / 2
    cx2 = l2 + w2 / 2
    cy2 = t2 + h2 / 2

    b_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    # If Ground, box distance undefined? Or large?
    # Let's set to -1 or 0?
    # If we set to 0, it looks like contact.
    # But ground doesn't have a box.
    # Let's set to a default large value or 0 with IoU=0.
    # Actually, let's use 0 but rely on IoU=0 and is_ground=1.
    b_dist = np.where(is_g, 0.0, b_dist)
    df[f"box_distance{out_suffix}"] = b_dist

    # Pass through raw visual features
    # P1
    df[f"p1_area{out_suffix}"] = area1
    df[f"p1_top{out_suffix}"] = t1
    df[f"p1_left{out_suffix}"] = l1
    df[f"p1_width{out_suffix}"] = w1
    df[f"p1_height{out_suffix}"] = h1

    # P2
    # If Ground, zero out P2 box features
    df[f"p2_area{out_suffix}"] = np.where(is_g, 0.0, area2)
    df[f"p2_top{out_suffix}"] = np.where(is_g, 0.0, t2)
    df[f"p2_left{out_suffix}"] = np.where(is_g, 0.0, l2)
    df[f"p2_width{out_suffix}"] = np.where(is_g, 0.0, w2)
    df[f"p2_height{out_suffix}"] = np.where(is_g, 0.0, h2)


def _generate_dataset(
    meta_path, tracking_path, helmets_path, scaler=None, fit_scaler=False
):
    """
    Core logic to merge and process the dataset.
    """
    print(f"Generating dataset from {meta_path}...")
    meta_df = pd.read_csv(meta_path)

    # Filter for debugging
    if config.TRAIN_PARAMS["debug_sample_size"] is not None:
        print(f"DEBUG: Sampling {config.TRAIN_PARAMS['debug_sample_size']} rows...")
        meta_df = meta_df.sample(
            n=min(len(meta_df), config.TRAIN_PARAMS["debug_sample_size"]),
            random_state=config.SEED,
        ).copy()

    # Identify unique game_plays needed
    unique_gps = meta_df["game_play"].unique()

    # Process Tracking
    track_df = _process_tracking_entity_first(tracking_path, unique_gps)

    # Process Helmets
    helm_df = _process_helmets_max_pooling(helmets_path, unique_gps)

    # Prepare Metadata for Merge
    # Ensure IDs are strings
    meta_df["nfl_player_id_1"] = meta_df["nfl_player_id_1"].astype(str)
    meta_df["nfl_player_id_2"] = meta_df["nfl_player_id_2"].astype(str)

    # Flag Ground
    meta_df["is_ground"] = (meta_df["nfl_player_id_2"] == "G").astype(int)

    # Merge P1 Tracking
    print("Merging P1 tracking...")
    # We merge on [game_play, nfl_player_id, step]
    # track_df has columns: game_play, nfl_player_id, step, x_position_lag_-5, ...

    # Rename track_df columns for P1 merge to avoid collision later?
    # Actually, let's merge and then rename via suffix
    merged = meta_df.merge(
        track_df,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(
        columns=["nfl_player_id"]
    )  # drop the join key from right

    # Rename lag columns to have _1 suffix
    # Current lag cols: {col}_lag_{k}
    # We want: {col}_lag_{k}_1
    rename_map_1 = {
        f"{c}_lag_{k}": f"{c}_lag_{k}_1"
        for c in RAW_TRACKING_COLS
        for k in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1)
    }
    merged = merged.rename(columns=rename_map_1)

    # Merge P2 Tracking
    print("Merging P2 tracking...")
    # Note: P2 can be 'G'. 'G' is not in track_df. Merge will result in NaNs for P2 cols.
    # We handle this in _compute_derived_features using 'is_ground' flag.
    merged = merged.merge(
        track_df,
        left_on=["game_play", "nfl_player_id_2", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    rename_map_2 = {
        f"{c}_lag_{k}": f"{c}_lag_{k}_2"
        for c in RAW_TRACKING_COLS
        for k in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1)
    }
    merged = merged.rename(columns=rename_map_2)

    # Merge P1 Helmets
    print("Merging P1 helmets...")
    merged = merged.merge(
        helm_df,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    rename_map_h1 = {
        f"{c}_lag_{k}": f"{c}_lag_{k}_1"
        for c in RAW_HELMET_COLS + ["area"]
        for k in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1)
    }
    merged = merged.rename(columns=rename_map_h1)

    # Merge P2 Helmets
    print("Merging P2 helmets...")
    merged = merged.merge(
        helm_df,
        left_on=["game_play", "nfl_player_id_2", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    rename_map_h2 = {
        f"{c}_lag_{k}": f"{c}_lag_{k}_2"
        for c in RAW_HELMET_COLS + ["area"]
        for k in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1)
    }
    merged = merged.rename(columns=rename_map_h2)

    # Compute Features per Lag
    print("Computing derived features...")
    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        # Suffixes in merged df: _lag_{k}_1 and _lag_{k}_2
        suffix_p1 = f"_lag_{lag}_1"
        suffix_p2 = f"_lag_{lag}_2"
        # Output suffix for final features: just _lag_{k} (we will flatten later)
        # But wait, config.KINEMATIC_FEATURES names don't have lag suffix.
        # We need to construct the final list.
        # Let's append _lag_{k} to the feature names in the dataframe.
        out_suffix = f"_lag_{lag}"

        _compute_derived_features(merged, suffix_p1, suffix_p2, out_suffix)

    # Select Final Columns
    # Kinematic
    kin_cols = []
    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        for feat in config.KINEMATIC_FEATURES:
            kin_cols.append(f"{feat}_lag_{lag}")

    # Visual
    vis_cols = []
    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        for feat in config.VISUAL_FEATURES:
            vis_cols.append(f"{feat}_lag_{lag}")

    # Fill remaining NaNs (e.g. missing tracking/helmet data not covered by Ground logic)
    # Fill with 0 is standard for NN input when signal is missing
    print("Imputing missing values...")
    merged[kin_cols] = merged[kin_cols].fillna(0.0)
    merged[vis_cols] = merged[vis_cols].fillna(0.0)

    X_kin = merged[kin_cols].values.astype(np.float32)
    X_vis = merged[vis_cols].values.astype(np.float32)
    y = merged["contact"].values.astype(np.float32)
    ids = merged["contact_id"].values

    # Scaling
    if fit_scaler:
        print("Fitting scaler...")
        # Scale Kinematic and Visual separately? Or together?
        # Usually separately or one big scaler.
        # Let's use one big scaler for simplicity, concatenating then splitting?
        # No, better to have separate scalers or just one if we concat.
        # The model treats them as separate streams.
        # Let's fit one scaler on concatenated features to handle relative magnitudes if needed,
        # but here features are distinct.
        # Let's fit one scaler on everything for simplicity of management.
        scaler = StandardScaler()
        X_all = np.hstack([X_kin, X_vis])
        scaler.fit(X_all)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)

    if scaler is not None:
        print("Applying scaler...")
        X_all = np.hstack([X_kin, X_vis])
        X_all = scaler.transform(X_all)
        # Split back
        X_kin = X_all[:, : X_kin.shape[1]]
        X_vis = X_all[:, X_kin.shape[1] :]

    return X_kin, X_vis, y, ids, scaler


# =============================================================================
# PUBLIC API
# =============================================================================


def get_train_data(load_cached_data=True):
    cache_kin = os.path.join(config.WORKING_DIR, "train_X_kin.npy")
    cache_vis = os.path.join(config.WORKING_DIR, "train_X_vis.npy")
    cache_y = os.path.join(config.WORKING_DIR, "train_y.npy")
    cache_ids = os.path.join(config.WORKING_DIR, "train_ids.npy")

    if (
        load_cached_data
        and os.path.exists(cache_kin)
        and os.path.exists(config.SCALER_SAVE_PATH)
    ):
        print("Loading cached train data...")
        return (
            np.load(cache_kin),
            np.load(cache_vis),
            np.load(cache_y),
            np.load(cache_ids, allow_pickle=True),
        )

    X_kin, X_vis, y, ids, _ = _generate_dataset(
        config.META_TRAIN_PATH,
        config.TRAIN_TRACKING_PATH,
        config.TRAIN_HELMETS_PATH,
        fit_scaler=True,
    )

    # Cache
    np.save(cache_kin, X_kin)
    np.save(cache_vis, X_vis)
    np.save(cache_y, y)
    np.save(cache_ids, ids)

    return X_kin, X_vis, y, ids


def get_val_data(load_cached_data=True):
    cache_kin = os.path.join(config.WORKING_DIR, "val_X_kin.npy")
    cache_vis = os.path.join(config.WORKING_DIR, "val_X_vis.npy")
    cache_y = os.path.join(config.WORKING_DIR, "val_y.npy")
    cache_ids = os.path.join(config.WORKING_DIR, "val_ids.npy")

    if load_cached_data and os.path.exists(cache_kin):
        print("Loading cached validation data...")
        return (
            np.load(cache_kin),
            np.load(cache_vis),
            np.load(cache_y),
            np.load(cache_ids, allow_pickle=True),
        )

    scaler = joblib.load(config.SCALER_SAVE_PATH)
    X_kin, X_vis, y, ids, _ = _generate_dataset(
        config.META_VAL_PATH,
        config.TRAIN_TRACKING_PATH,  # Val is subset of Train files
        config.TRAIN_HELMETS_PATH,
        scaler=scaler,
        fit_scaler=False,
    )

    np.save(cache_kin, X_kin)
    np.save(cache_vis, X_vis)
    np.save(cache_y, y)
    np.save(cache_ids, ids)

    return X_kin, X_vis, y, ids


def get_test_data(load_cached_data=True):
    cache_kin = os.path.join(config.WORKING_DIR, "test_X_kin.npy")
    cache_vis = os.path.join(config.WORKING_DIR, "test_X_vis.npy")
    cache_y = os.path.join(config.WORKING_DIR, "test_y.npy")
    cache_ids = os.path.join(config.WORKING_DIR, "test_ids.npy")

    if load_cached_data and os.path.exists(cache_kin):
        print("Loading cached test data...")
        return (
            np.load(cache_kin),
            np.load(cache_vis),
            np.load(cache_y),
            np.load(cache_ids, allow_pickle=True),
        )

    scaler = joblib.load(config.SCALER_SAVE_PATH)
    X_kin, X_vis, y, ids, _ = _generate_dataset(
        config.META_TEST_PATH,
        config.TEST_TRACKING_PATH,
        config.TEST_HELMETS_PATH,
        scaler=scaler,
        fit_scaler=False,
    )

    np.save(cache_kin, X_kin)
    np.save(cache_vis, X_vis)
    np.save(cache_y, y)
    np.save(cache_ids, ids)

    return X_kin, X_vis, y, ids


def save_submission(ids, probs, threshold=0.5):
    """
    Generates submission file.
    """
    print(f"Generating submission with threshold {threshold}...")
    preds = (probs >= threshold).astype(int)

    df = pd.DataFrame({"contact_id": ids, "contact": preds})

    # Ensure all sample_submission IDs are present
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Merge to ensure order and completeness
    # Left join sample on preds
    submission = sample[["contact_id"]].merge(df, on="contact_id", how="left")

    # Fill missing with 0
    submission["contact"] = submission["contact"].fillna(0).astype(int)

    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
