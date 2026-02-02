import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config


def calculate_iou(box1, box2):
    """
    Vectorized IoU calculation.
    box: [left, top, width, height]
    """
    b1_x1, b1_y1 = box1[:, 0], box1[:, 1]
    b1_x2, b1_y2 = b1_x1 + box1[:, 2], b1_y1 + box1[:, 3]

    b2_x1, b2_y1 = box2[:, 0], box2[:, 1]
    b2_x2, b2_y2 = b2_x1 + box2[:, 2], b2_y1 + box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    b1_area = box1[:, 2] * box1[:, 3]
    b2_area = box2[:, 2] * box2[:, 3]

    union_area = b1_area + b2_area - inter_area

    return np.divide(
        inter_area, union_area, out=np.zeros_like(inter_area), where=union_area != 0
    )


def calculate_centroid_distance(box1, box2):
    """
    Vectorized Euclidean distance between box centers.
    """
    c1_x = box1[:, 0] + box1[:, 2] / 2
    c1_y = box1[:, 1] + box1[:, 3] / 2

    c2_x = box2[:, 0] + box2[:, 2] / 2
    c2_y = box2[:, 1] + box2[:, 3] / 2

    return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)


def process_data(split_name: str, load_cached_data: bool = True):
    """
    Main data processing pipeline implementing the SRV-Net data strategy.

    Args:
        split_name: 'train', 'validation', or 'test'
        load_cached_data: If True, attempts to load .npy files from disk.

    Returns:
        X_kin (np.array): Kinematic features [N, Features]
        X_vis (np.array): Stereoscopic visual features [N, Features]
        y (np.array): Targets [N]
        ids (np.array): Contact IDs [N]
    """
    # Setup paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_X_kin = os.path.join(cache_dir, f"{split_name}_X_kin.npy")
    path_X_vis = os.path.join(cache_dir, f"{split_name}_X_vis.npy")
    path_y = os.path.join(cache_dir, f"{split_name}_y.npy")
    path_ids = os.path.join(cache_dir, f"{split_name}_ids.npy")
    path_scaler = os.path.join(cache_dir, "scaler.joblib")

    # 1. Cache Check
    if load_cached_data:
        if all(os.path.exists(p) for p in [path_X_kin, path_X_vis, path_y, path_ids]):
            print(f"Loading {split_name} data from cache...")
            return (
                np.load(path_X_kin),
                np.load(path_X_vis),
                np.load(path_y),
                np.load(path_ids, allow_pickle=True),
            )

    print(f"Processing {split_name} data from scratch...")

    # 2. Load Raw Data
    meta_path = os.path.join(Config.METADATA_DIR, f"{split_name}.csv")
    df_meta = pd.read_csv(meta_path)

    if split_name == "test":
        track_path = os.path.join(Config.INPUT_DIR, "test_player_tracking.csv")
        helmet_path = os.path.join(Config.INPUT_DIR, "test_baseline_helmets.csv")
    else:
        track_path = os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
        helmet_path = os.path.join(Config.INPUT_DIR, "train_baseline_helmets.csv")

    df_track = pd.read_csv(track_path)
    df_helmets = pd.read_csv(helmet_path)

    # Filter for memory efficiency
    relevant_gps = df_meta["game_play"].unique()
    df_track = df_track[df_track["game_play"].isin(relevant_gps)].copy()
    df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_gps)].copy()

    # 3. Process Tracking (Entity-First)
    track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]
    df_track = df_track[track_cols]

    # 4. Process Helmets (Stereoscopic Pivot)
    # Map frame to step (approximate based on 59.94Hz video and 10Hz tracking)
    # Snap is ~frame 300. Step 0 is snap.
    df_helmets["step"] = np.round((df_helmets["frame"] - 300) / 5.994).astype(int)

    df_helmets["view_code"] = df_helmets["view"].map(
        {"Sideline": "side", "Endzone": "end"}
    )

    # Deduplicate and Pivot
    df_helmets = (
        df_helmets.groupby(["game_play", "step", "nfl_player_id", "view_code"])
        .first()
        .reset_index()
    )
    h_pivot = df_helmets.pivot_table(
        index=["game_play", "step", "nfl_player_id"],
        columns="view_code",
        values=["left", "top", "width", "height"],
        aggfunc="first",
    )
    h_pivot.columns = [f"{c[1]}_{c[0]}" for c in h_pivot.columns]
    h_pivot = h_pivot.reset_index()

    # 5. Merge Streams
    # Prepare Metadata
    df_meta["nfl_player_id_1"] = (
        pd.to_numeric(df_meta["nfl_player_id_1"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    df_meta["is_ground"] = (df_meta["nfl_player_id_2"] == "G").astype(int)
    df_meta["nfl_player_id_2_int"] = (
        pd.to_numeric(df_meta["nfl_player_id_2"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Merge Tracking P1
    df_merged = df_meta.merge(
        df_track.add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
    )

    # Merge Tracking P2
    df_merged = df_merged.merge(
        df_track.add_suffix("_2"),
        left_on=["game_play", "step", "nfl_player_id_2_int"],
        right_on=["game_play_2", "step_2", "nfl_player_id_2"],
        how="left",
    )

    # Ground Imputation (Tracking)
    mask_g = df_merged["is_ground"] == 1
    df_merged.loc[mask_g, "x_position_2"] = df_merged.loc[mask_g, "x_position_1"]
    df_merged.loc[mask_g, "y_position_2"] = df_merged.loc[mask_g, "y_position_1"]
    for c in ["speed_2", "acceleration_2", "orientation_2", "direction_2", "sa_2"]:
        df_merged.loc[mask_g, c] = 0.0

    # Fill missing tracking
    track_feat_cols = [
        c for c in df_merged.columns if c.endswith("_1") or c.endswith("_2")
    ]
    df_merged[track_feat_cols] = df_merged[track_feat_cols].fillna(0.0)

    # Relative Kinematics
    dx = df_merged["x_position_1"] - df_merged["x_position_2"]
    dy = df_merged["y_position_1"] - df_merged["y_position_2"]
    dist = np.sqrt(dx**2 + dy**2)
    df_merged["distance"] = dist
    df_merged["log_distance"] = np.log1p(dist)
    df_merged["relative_speed"] = df_merged["speed_1"] - df_merged["speed_2"]

    # Closing Speed
    def get_v(s, d):
        rad = np.radians(d)
        return s * np.sin(rad), s * np.cos(rad)

    vx1, vy1 = get_v(df_merged["speed_1"], df_merged["direction_1"])
    vx2, vy2 = get_v(df_merged["speed_2"], df_merged["direction_2"])
    dvx, dvy = vx1 - vx2, vy1 - vy2
    dot_prod = dvx * dx + dvy * dy
    df_merged["closing_speed"] = -(
        np.divide(dot_prod, dist, out=np.zeros_like(dot_prod), where=dist != 0)
    )

    # Merge Helmets
    df_merged = df_merged.merge(
        h_pivot.add_suffix("_p1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
        how="left",
    )
    df_merged = df_merged.merge(
        h_pivot.add_suffix("_p2"),
        left_on=["game_play", "step", "nfl_player_id_2_int"],
        right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
        how="left",
    )

    # Fill missing helmets
    h_cols = [
        c
        for c in df_merged.columns
        if ("_p1" in c or "_p2" in c) and ("side_" in c or "end_" in c)
    ]
    df_merged[h_cols] = df_merged[h_cols].fillna(0.0)

    # Visual Features (Stereoscopic)
    # Sideline
    box_s1 = df_merged[
        ["side_left_p1", "side_top_p1", "side_width_p1", "side_height_p1"]
    ].values
    box_s2 = df_merged[
        ["side_left_p2", "side_top_p2", "side_width_p2", "side_height_p2"]
    ].values
    df_merged["sideline_iou"] = calculate_iou(box_s1, box_s2)
    df_merged["sideline_dist"] = calculate_centroid_distance(box_s1, box_s2)
    df_merged["sideline_p1_top"] = df_merged["side_top_p1"]
    df_merged["sideline_p2_top"] = df_merged["side_top_p2"]
    df_merged["sideline_p1_area"] = (
        df_merged["side_width_p1"] * df_merged["side_height_p1"]
    )
    df_merged["sideline_p2_area"] = (
        df_merged["side_width_p2"] * df_merged["side_height_p2"]
    )
    df_merged["sideline_avail"] = (df_merged["side_width_p1"] > 0).astype(float)

    # Endzone
    box_e1 = df_merged[
        ["end_left_p1", "end_top_p1", "end_width_p1", "end_height_p1"]
    ].values
    box_e2 = df_merged[
        ["end_left_p2", "end_top_p2", "end_width_p2", "end_height_p2"]
    ].values
    df_merged["endzone_iou"] = calculate_iou(box_e1, box_e2)
    df_merged["endzone_dist"] = calculate_centroid_distance(box_e1, box_e2)
    df_merged["endzone_p1_top"] = df_merged["end_top_p1"]
    df_merged["endzone_p2_top"] = df_merged["end_top_p2"]
    df_merged["endzone_p1_area"] = (
        df_merged["end_width_p1"] * df_merged["end_height_p1"]
    )
    df_merged["endzone_p2_area"] = (
        df_merged["end_width_p2"] * df_merged["end_height_p2"]
    )
    df_merged["endzone_avail"] = (df_merged["end_width_p1"] > 0).astype(float)

    # 6. Temporal Windowing
    df_merged = df_merged.sort_values(
        ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
    )
    grp = df_merged.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

    kin_feats = Config.KINEMATIC_FEATURES_SINGLE_STEP
    vis_feats = Config.VISUAL_FEATURES_SINGLE_STEP

    kin_cols_out = []
    vis_cols_out = []

    # Collect shifted dataframes
    shifted_dfs = []

    for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_t{offset:+d}" if offset != 0 else "_t0"

        # Kinematic
        s_kin = grp[kin_feats].shift(-offset).fillna(0)
        s_kin.columns = [f"{c}{suffix}" for c in kin_feats]
        shifted_dfs.append(s_kin)
        kin_cols_out.extend(s_kin.columns)

        # Visual
        s_vis = grp[vis_feats].shift(-offset).fillna(0)
        s_vis.columns = [f"{c}{suffix}" for c in vis_feats]
        shifted_dfs.append(s_vis)
        vis_cols_out.extend(s_vis.columns)

    df_final = pd.concat([df_merged] + shifted_dfs, axis=1)

    # 7. Normalization & Saving
    X_kin = df_final[kin_cols_out].values.astype(np.float32)
    X_vis = df_final[vis_cols_out].values.astype(np.float32)
    y = df_final["contact"].values.astype(np.float32)
    ids = df_final["contact_id"].values

    if split_name == "train":
        scaler_kin = StandardScaler()
        scaler_vis = StandardScaler()
        X_kin = scaler_kin.fit_transform(X_kin)
        X_vis = scaler_vis.fit_transform(X_vis)
        joblib.dump((scaler_kin, scaler_vis), path_scaler)
    else:
        if os.path.exists(path_scaler):
            scaler_kin, scaler_vis = joblib.load(path_scaler)
            X_kin = scaler_kin.transform(X_kin)
            X_vis = scaler_vis.transform(X_vis)
        else:
            print("Warning: Scaler not found. Skipping normalization.")

    print(f"Saving {split_name} data...")
    np.save(path_X_kin, X_kin)
    np.save(path_X_vis, X_vis)
    np.save(path_y, y)
    np.save(path_ids, ids)

    return X_kin, X_vis, y, ids
