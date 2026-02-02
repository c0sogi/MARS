import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


def compute_iou(box1, box2):
    """
    Vectorized IoU calculation.
    box: [left, top, width, height]
    """
    l1, t1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    l2, t2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    r1 = l1 + w1
    b1 = t1 + h1
    r2 = l2 + w2
    b2 = t2 + h2

    x_left = np.maximum(l1, l2)
    y_top = np.maximum(t1, t2)
    x_right = np.minimum(r1, r2)
    y_bottom = np.minimum(b1, b2)

    intersection_w = np.maximum(0, x_right - x_left)
    intersection_h = np.maximum(0, y_bottom - y_top)
    intersection_area = intersection_w * intersection_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    iou = np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(intersection_area),
        where=union_area > 0,
    )
    return iou


def compute_centroid_distance(box1, box2):
    """
    Vectorized centroid distance.
    """
    l1, t1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    l2, t2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    cx1 = l1 + w1 / 2
    cy1 = t1 + h1 / 2
    cx2 = l2 + w2 / 2
    cy2 = t2 + h2 / 2

    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist


def preprocess_tracking(df_tracking):
    """
    Process tracking data: Filter, Sort, Window.
    Generates wide feature vector for t-K to t+K.
    """
    cols = ["game_play", "nfl_player_id", "step"] + Config.TRACKING_RAW_COLS
    df = df_tracking[cols].copy()

    # Sort for correct shifting
    df = df.sort_values(["game_play", "nfl_player_id", "step"])

    feature_cols = Config.TRACKING_RAW_COLS
    grouped = df.groupby(["game_play", "nfl_player_id"])

    lagged_data = {}
    for col in feature_cols:
        for k in range(-Config.WINDOW_K, Config.WINDOW_K + 1):
            # shift(-k) gets value at t+k
            lagged_data[f"{col}_lag_{k}"] = grouped[col].shift(-k)

    df_lags = pd.DataFrame(lagged_data, index=df.index)

    # Combine key columns with lagged features
    df_out = pd.concat([df[["game_play", "nfl_player_id", "step"]], df_lags], axis=1)

    # Fill NaNs at edges of play with 0
    df_out = df_out.fillna(0)

    return df_out


def preprocess_helmets(df_helmets):
    """
    Process helmet data: Map Frame->Step, Deduplicate Views, Window.
    """
    cols = ["game_play", "nfl_player_id", "frame"] + Config.HELMET_RAW_COLS
    df = df_helmets[cols].copy()

    # Map Frame to Step
    # Step 0 = Snap = 5s = Frame 300. Tracking 10Hz, Video 59.94Hz.
    # frame = 300 + step * 5.994 => step = (frame - 300) / 5.994
    df["step"] = ((df["frame"] - 300) / 5.994).round().astype(int)

    # Deduplicate: Handle multiple views (Sideline/Endzone) by keeping the largest box
    df["area"] = df["width"] * df["height"]
    df = df.sort_values(
        ["game_play", "nfl_player_id", "step", "area"],
        ascending=[True, True, True, False],
    )
    df = df.drop_duplicates(subset=["game_play", "nfl_player_id", "step"], keep="first")

    # Windowing
    feature_cols = Config.HELMET_RAW_COLS
    grouped = df.groupby(["game_play", "nfl_player_id"])

    lagged_data = {}
    for col in feature_cols:
        for k in range(-Config.WINDOW_K, Config.WINDOW_K + 1):
            lagged_data[f"{col}_lag_{k}"] = grouped[col].shift(-k)

    df_lags = pd.DataFrame(lagged_data, index=df.index)

    df_out = pd.concat([df[["game_play", "nfl_player_id", "step"]], df_lags], axis=1)
    df_out = df_out.fillna(0)

    return df_out


def merge_and_engineer(df_meta, df_track_proc, df_helmet_proc):
    """
    Merges processed tracking/helmet data onto labels and computes pair features.
    Handles Hybrid Ground Imputation.
    """
    # Ensure merge keys are numeric
    df_meta["nfl_player_id_1"] = pd.to_numeric(
        df_meta["nfl_player_id_1"], errors="coerce"
    )
    df_meta["nfl_player_id_2_num"] = pd.to_numeric(
        df_meta["nfl_player_id_2"], errors="coerce"
    )

    # --- Merge Player 1 ---
    # Tracking
    df_merged = df_meta.merge(
        df_track_proc,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_drop"),
    ).drop(columns=["nfl_player_id", "step_drop"], errors="ignore")

    p1_track_cols = [c for c in df_track_proc.columns if "lag" in c]
    df_merged = df_merged.rename(columns={c: f"{c}_p1" for c in p1_track_cols})

    # Helmets
    df_merged = df_merged.merge(
        df_helmet_proc,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"], errors="ignore")

    p1_helmet_cols = [c for c in df_helmet_proc.columns if "lag" in c]
    df_merged = df_merged.rename(columns={c: f"{c}_p1" for c in p1_helmet_cols})

    # --- Merge Player 2 ---
    # Tracking
    df_merged = df_merged.merge(
        df_track_proc,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"], errors="ignore")

    p2_track_cols = [c for c in df_track_proc.columns if "lag" in c]
    df_merged = df_merged.rename(columns={c: f"{c}_p2" for c in p2_track_cols})

    # Helmets
    df_merged = df_merged.merge(
        df_helmet_proc,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"], errors="ignore")

    p2_helmet_cols = [c for c in df_helmet_proc.columns if "lag" in c]
    df_merged = df_merged.rename(columns={c: f"{c}_p2" for c in p2_helmet_cols})

    # --- Ground Imputation ---
    is_ground = df_merged["nfl_player_id_2"] == "G"

    for k in range(-Config.WINDOW_K, Config.WINDOW_K + 1):
        suffix = f"_lag_{k}"
        # Kinematics: P2 Pos = P1 Pos, Velocity = 0
        df_merged.loc[is_ground, f"x_position{suffix}_p2"] = df_merged.loc[
            is_ground, f"x_position{suffix}_p1"
        ]
        df_merged.loc[is_ground, f"y_position{suffix}_p2"] = df_merged.loc[
            is_ground, f"y_position{suffix}_p1"
        ]

        for col in ["speed", "acceleration", "sa", "direction", "orientation"]:
            df_merged.loc[is_ground, f"{col}{suffix}_p2"] = 0

    # Visuals: P2 Box -> 0
    for col in p2_helmet_cols:
        c_name = f"{col}_p2"
        if c_name in df_merged.columns:
            df_merged.loc[is_ground, c_name] = 0

    df_merged = df_merged.fillna(0)

    # --- Compute Pair Features ---
    for k in range(-Config.WINDOW_K, Config.WINDOW_K + 1):
        suffix = f"_lag_{k}"

        # Kinematic Pair
        x1 = df_merged[f"x_position{suffix}_p1"]
        y1 = df_merged[f"y_position{suffix}_p1"]
        x2 = df_merged[f"x_position{suffix}_p2"]
        y2 = df_merged[f"y_position{suffix}_p2"]

        dx = x1 - x2
        dy = y1 - y2
        dist = np.sqrt(dx**2 + dy**2)

        df_merged[f"distance{suffix}"] = dist
        df_merged[f"log_distance{suffix}"] = np.log1p(dist)

        s1 = df_merged[f"speed{suffix}_p1"]
        s2 = df_merged[f"speed{suffix}_p2"]
        df_merged[f"relative_speed{suffix}"] = np.abs(s1 - s2)

        # Relative Acceleration (Cite Lesson 00033)
        a1 = df_merged[f"acceleration{suffix}_p1"]
        a2 = df_merged[f"acceleration{suffix}_p2"]
        df_merged[f"relative_acceleration{suffix}"] = np.abs(a1 - a2)

        # Orientation Difference (Cite Lesson 00033)
        o1 = df_merged[f"orientation{suffix}_p1"]
        o2 = df_merged[f"orientation{suffix}_p2"]
        diff = np.abs(o1 - o2)
        df_merged[f"orientation_diff{suffix}"] = np.minimum(diff, 360 - diff)

        # Closing Speed
        d1 = np.radians(df_merged[f"direction{suffix}_p1"])
        d2 = np.radians(df_merged[f"direction{suffix}_p2"])

        vx1 = s1 * np.sin(d1)
        vy1 = s1 * np.cos(d1)
        vx2 = s2 * np.sin(d2)
        vy2 = s2 * np.cos(d2)

        rvx = vx1 - vx2
        rvy = vy1 - vy2

        dot = rvx * dx + rvy * dy
        # Use max(dist, eps) for numerical stability (Cite Lesson 00007)
        closing = -(dot) / np.maximum(dist, 1e-6)
        df_merged[f"clamped_closing_speed{suffix}"] = np.clip(closing, -10, 30)

        # Visual Pair
        b1 = df_merged[
            [
                f"left{suffix}_p1",
                f"top{suffix}_p1",
                f"width{suffix}_p1",
                f"height{suffix}_p1",
            ]
        ].values
        b2 = df_merged[
            [
                f"left{suffix}_p2",
                f"top{suffix}_p2",
                f"width{suffix}_p2",
                f"height{suffix}_p2",
            ]
        ].values

        df_merged[f"helmet_iou{suffix}"] = compute_iou(b1, b2)
        h_dist = compute_centroid_distance(b1, b2)
        df_merged[f"helmet_centroid_dist{suffix}"] = h_dist
        # Log transform for visual distance (Cite Lesson 00005)
        df_merged[f"log_helmet_centroid_dist{suffix}"] = np.log1p(h_dist)

    return df_merged


def create_datasets(load_cached_data=True):
    """
    Main entry point. Loads, processes, scales, and returns dataset arrays.
    """
    cache_dir = Config.WORKING_DIR
    files = [
        "train_X_kin.npy",
        "train_X_vis.npy",
        "train_y.npy",
        "train_ids.npy",
        "val_X_kin.npy",
        "val_X_vis.npy",
        "val_y.npy",
        "val_ids.npy",
        "test_X_kin.npy",
        "test_X_vis.npy",
        "test_ids.npy",
        "test_df.parquet",
    ]

    cache_exists = all([os.path.exists(os.path.join(cache_dir, f)) for f in files])

    if load_cached_data and cache_exists:
        print("Loading cached datasets...")
        train_X_kin = np.load(os.path.join(cache_dir, "train_X_kin.npy"))
        train_X_vis = np.load(os.path.join(cache_dir, "train_X_vis.npy"))
        train_y = np.load(os.path.join(cache_dir, "train_y.npy"))
        train_ids = np.load(os.path.join(cache_dir, "train_ids.npy"), allow_pickle=True)

        val_X_kin = np.load(os.path.join(cache_dir, "val_X_kin.npy"))
        val_X_vis = np.load(os.path.join(cache_dir, "val_X_vis.npy"))
        val_y = np.load(os.path.join(cache_dir, "val_y.npy"))
        val_ids = np.load(os.path.join(cache_dir, "val_ids.npy"), allow_pickle=True)

        test_X_kin = np.load(os.path.join(cache_dir, "test_X_kin.npy"))
        test_X_vis = np.load(os.path.join(cache_dir, "test_X_vis.npy"))
        test_ids = np.load(os.path.join(cache_dir, "test_ids.npy"), allow_pickle=True)
        test_df = pd.read_parquet(os.path.join(cache_dir, "test_df.parquet"))

        return (
            (train_X_kin, train_X_vis, train_y, train_ids),
            (val_X_kin, val_X_vis, val_y, val_ids),
            (test_X_kin, test_X_vis, test_ids, test_df),
        )

    print("Processing data from scratch...")

    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    train_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    test_tracking = pd.read_csv(Config.TEST_TRACKING_PATH)

    train_helmets = pd.read_csv(Config.TRAIN_HELMETS_PATH)
    test_helmets = pd.read_csv(Config.TEST_HELMETS_PATH)

    print("Processing Tracking...")
    train_track_proc = preprocess_tracking(train_tracking)
    test_track_proc = preprocess_tracking(test_tracking)

    print("Processing Helmets...")
    train_helmet_proc = preprocess_helmets(train_helmets)
    test_helmet_proc = preprocess_helmets(test_helmets)

    print("Merging Datasets...")
    train_merged = merge_and_engineer(train_meta, train_track_proc, train_helmet_proc)
    val_merged = merge_and_engineer(val_meta, train_track_proc, train_helmet_proc)
    test_merged = merge_and_engineer(test_meta, test_track_proc, test_helmet_proc)

    # Define Column Order
    kin_cols = []
    vis_cols = []

    for k in range(-Config.WINDOW_K, Config.WINDOW_K + 1):
        suffix = f"_lag_{k}"
        for c in Config.KINEMATIC_PLAYER_COLS:
            kin_cols.append(f"{c}{suffix}_p1")
        for c in Config.KINEMATIC_PLAYER_COLS:
            kin_cols.append(f"{c}{suffix}_p2")
        for c in Config.KINEMATIC_PAIR_COLS:
            kin_cols.append(f"{c}{suffix}")

        for c in Config.VISUAL_PLAYER_COLS:
            vis_cols.append(f"{c}{suffix}_p1")
        for c in Config.VISUAL_PLAYER_COLS:
            vis_cols.append(f"{c}{suffix}_p2")
        for c in Config.VISUAL_PAIR_COLS:
            vis_cols.append(f"{c}{suffix}")

    print("Scaling...")
    scaler_kin = StandardScaler()
    scaler_vis = StandardScaler()

    X_kin_train = scaler_kin.fit_transform(
        train_merged[kin_cols].values.astype(np.float32)
    )
    X_vis_train = scaler_vis.fit_transform(
        train_merged[vis_cols].values.astype(np.float32)
    )

    X_kin_val = scaler_kin.transform(val_merged[kin_cols].values.astype(np.float32))
    X_vis_val = scaler_vis.transform(val_merged[vis_cols].values.astype(np.float32))

    X_kin_test = scaler_kin.transform(test_merged[kin_cols].values.astype(np.float32))
    X_vis_test = scaler_vis.transform(test_merged[vis_cols].values.astype(np.float32))

    joblib.dump({"kin": scaler_kin, "vis": scaler_vis}, Config.SCALER_PATH)

    train_y = train_merged["contact"].values.astype(np.float32)
    train_ids = train_merged["contact_id"].values
    val_y = val_merged["contact"].values.astype(np.float32)
    val_ids = val_merged["contact_id"].values
    test_ids = test_merged["contact_id"].values

    print("Caching...")
    np.save(os.path.join(cache_dir, "train_X_kin.npy"), X_kin_train)
    np.save(os.path.join(cache_dir, "train_X_vis.npy"), X_vis_train)
    np.save(os.path.join(cache_dir, "train_y.npy"), train_y)
    np.save(os.path.join(cache_dir, "train_ids.npy"), train_ids)

    np.save(os.path.join(cache_dir, "val_X_kin.npy"), X_kin_val)
    np.save(os.path.join(cache_dir, "val_X_vis.npy"), X_vis_val)
    np.save(os.path.join(cache_dir, "val_y.npy"), val_y)
    np.save(os.path.join(cache_dir, "val_ids.npy"), val_ids)

    np.save(os.path.join(cache_dir, "test_X_kin.npy"), X_kin_test)
    np.save(os.path.join(cache_dir, "test_X_vis.npy"), X_vis_test)
    np.save(os.path.join(cache_dir, "test_ids.npy"), test_ids)
    test_merged.to_parquet(os.path.join(cache_dir, "test_df.parquet"))

    return (
        (X_kin_train, X_vis_train, train_y, train_ids),
        (X_kin_val, X_vis_val, val_y, val_ids),
        (X_kin_test, X_vis_test, test_ids, test_merged),
    )
