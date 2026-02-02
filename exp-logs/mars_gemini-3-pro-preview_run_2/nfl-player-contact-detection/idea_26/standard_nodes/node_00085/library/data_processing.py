import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from library.config import Config
from library.utils import clamp_values, shortest_arc_distance

# =========================================================================
# Helper Functions
# =========================================================================


def _get_cache_path(filename):
    return os.path.join(Config.WORKING_DIR, filename)


def _load_or_create_encoders(df_tracking, fit=True):
    """
    Handles LabelEncoders for categorical features (position, team).
    """
    encoders = {}
    path_pos = _get_cache_path("encoder_position.joblib")
    path_team = _get_cache_path("encoder_team.joblib")

    if fit:
        # Fit and save
        le_pos = LabelEncoder()
        # Ensure all possible positions are covered or handle unknowns later
        # We fit on the provided dataframe (usually train tracking)
        le_pos.fit(df_tracking["position"].astype(str).fillna("Unknown"))

        le_team = LabelEncoder()
        le_team.fit(df_tracking["team"].astype(str).fillna("Unknown"))

        joblib.dump(le_pos, path_pos)
        joblib.dump(le_team, path_team)
        encoders["position"] = le_pos
        encoders["team"] = le_team
    else:
        # Load existing
        if os.path.exists(path_pos) and os.path.exists(path_team):
            encoders["position"] = joblib.load(path_pos)
            encoders["team"] = joblib.load(path_team)
        else:
            # Fallback if not found (should not happen if train runs first)
            # Create dummy encoders to prevent crash, but this implies logic error in pipeline order
            le_pos = LabelEncoder()
            le_pos.fit(df_tracking["position"].astype(str).fillna("Unknown"))
            le_team = LabelEncoder()
            le_team.fit(df_tracking["team"].astype(str).fillna("Unknown"))
            encoders["position"] = le_pos
            encoders["team"] = le_team

    return encoders


def _encode_categorical(df, encoders):
    """
    Applies label encoding to the dataframe.
    """
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen labels by mapping them to a default or first class
            # A simple way is to use a mask
            series = df[col].astype(str).fillna("Unknown")
            # Check for unseen
            unseen = ~series.isin(le.classes_)
            if unseen.any():
                # Map unseen to the first class (usually 0)
                # This is a simplification; ideally we'd have an 'Unknown' class
                df[col + "_enc"] = 0
                known_mask = ~unseen
                df.loc[known_mask, col + "_enc"] = le.transform(series[known_mask])
            else:
                df[col + "_enc"] = le.transform(series)
    return df


# =========================================================================
# Data Processing Logic
# =========================================================================


def preprocess_tracking_data(tracking_path, split_name, load_cached_data=True):
    """
    Loads tracking data, creates windowed features (t-5 to t+5), and enforces physical constraints.
    """
    cache_file = _get_cache_path(f"tracking_processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached tracking data for {split_name}...")
        return pd.read_parquet(cache_file)

    print(f"Processing tracking data for {split_name}...")
    df = pd.read_csv(tracking_path)

    # 1. Physical Constraints (Clamping)
    # Clamp continuous kinematic features
    for col in ["speed", "acceleration", "sa"]:
        if col in df.columns:
            df[col] = clamp_values(df[col].values)

    # Angular continuity is handled during interaction feature creation or here?
    # We'll leave raw orientation/direction as is (0-360) and handle deltas carefully.

    # 2. Categorical Encoding
    # We fit encoders only on training data split logic, but here we process file by file.
    # To be safe, we assume 'train' split fits the encoders.
    is_train = "train" in split_name
    encoders = _load_or_create_encoders(df, fit=is_train)
    df = _encode_categorical(df, encoders)

    # 3. Windowing (Wide Format)
    # We need features for t-5 to t+5.
    # Sort to ensure correct shifting
    df = df.sort_values(["game_play", "nfl_player_id", "step"])

    # Define features to window
    window_cols = Config.KINEMATIC_FEATURES

    # We will create a list of dataframes to concat (more efficient than repeated assignments)
    dfs_to_concat = [
        df[["game_play", "nfl_player_id", "step", "position_enc", "team_enc"]]
    ]

    # Group by player to respect boundaries
    grouped = df.groupby(["game_play", "nfl_player_id"])

    # Generate lags
    for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        # offset -5 means t-5. shift(5) brings t-5 to t.
        # We want the value at t-5 to be available at row t.
        # So we shift by +5 (positive shift pushes data down, so row t gets t-5? No.)
        # shift(1): row 1 gets row 0. So row t gets t-1.
        # shift(5): row t gets t-5.
        # shift(-5): row t gets t+5.

        shifted = grouped[window_cols].shift(offset)
        # Rename columns
        suffix = f"_t{offset:+d}" if offset != 0 else "_t0"
        shifted.columns = [f"{c}{suffix}" for c in window_cols]
        dfs_to_concat.append(shifted)

    # Concatenate all window features
    df_wide = pd.concat(dfs_to_concat, axis=1)

    # Drop rows with NaNs caused by shifting (start/end of play)
    # Or fill them? Given we want robust training, dropping edges is safer,
    # but for test we must predict all.
    # Strategy: Fill with nearest valid observation (ffill/bfill) within group.
    # However, concat broke the group structure.
    # Re-impute is expensive.
    # Simple approach: Fill NaNs with 0 (assuming silence) or carry over.
    # Given the "Input Clamping" and "Entity-Aware" nature, 0 is a safe neutral for velocity,
    # but bad for position.
    # Let's use ffill/bfill on the wide dataframe grouped by gameplay/player? No, too slow.
    # We will fill NaNs with 0 for now to ensure numerical stability.
    df_wide = df_wide.fillna(0.0)

    # Downcast to float32 to save memory
    fcols = df_wide.select_dtypes(include=["float64"]).columns
    df_wide[fcols] = df_wide[fcols].astype("float32")

    # Cache
    df_wide.to_parquet(cache_file)
    return df_wide


def preprocess_helmet_data(helmet_path, split_name, load_cached_data=True):
    """
    Loads helmet data and applies Max-Pooling to select the best view.
    """
    cache_file = _get_cache_path(f"helmets_processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached helmet data for {split_name}...")
        return pd.read_parquet(cache_file)

    print(f"Processing helmet data for {split_name}...")
    df = pd.read_csv(helmet_path)

    # Calculate View Area
    df["view_area"] = df["width"] * df["height"]

    # Max-Pooling Strategy: Select row with max view_area per (game_play, step, nfl_player_id)
    # Sort by area descending
    df = df.sort_values("view_area", ascending=False)

    # Drop duplicates keeping the first (largest area)
    # Note: Helmets data uses 'frame', need to map to 'step' if not present?
    # The baseline_helmets.csv has 'frame'. Tracking has 'step'.
    # We assume a mapping exists or we need to approximate.
    # Wait, the task description says "Sideline and Endzone video pairs are matched frame for frame".
    # But tracking is 10Hz. Video is 59.94Hz.
    # We need to map frame -> step.
    # Step 0 is at snap. Snap is 5s into video (300 frames).
    # Step increment is 0.1s (approx 6 frames).
    # Formula: frame = 300 + step * 6 (approx).
    # Inverse: step = round((frame - 300) / 5.994).
    # Let's compute step from frame.

    df["step"] = np.round((df["frame"] - 300) / 5.994).astype(int)

    # Now group and max pool
    cols_to_keep = ["game_play", "step", "nfl_player_id"] + Config.VISUAL_FEATURES
    df_pooled = df.drop_duplicates(
        subset=["game_play", "step", "nfl_player_id"], keep="first"
    )[cols_to_keep]

    # Cache
    df_pooled.to_parquet(cache_file)
    return df_pooled


def create_features(
    metadata_path, tracking_path, helmet_path, split_name, load_cached_data=True
):
    """
    Merges metadata with tracking and visual features, handles ground imputation, and scales data.
    """
    final_cache_path = _get_cache_path(f"features_{split_name}.parquet")

    if load_cached_data and os.path.exists(final_cache_path):
        print(f"Loading final features for {split_name}...")
        return pd.read_parquet(final_cache_path)

    print(f"Creating features for {split_name}...")

    # 1. Load Metadata (Labels or Submission)
    df_meta = pd.read_csv(metadata_path)

    # 2. Load Preprocessed Inputs
    df_track = preprocess_tracking_data(tracking_path, split_name, load_cached_data)
    df_helm = preprocess_helmet_data(helmet_path, split_name, load_cached_data)

    # 3. Merge Player 1
    # Tracking
    df_merged = df_meta.merge(
        df_track,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    ).drop(
        columns=["nfl_player_id"]
    )  # Drop duplicate key

    # Visuals
    # Note: nfl_player_id_1 is numeric in tracking but might be string in metadata?
    # Metadata schema says nfl_player_id_1 is int/str. Ensure consistency.
    df_merged["nfl_player_id_1"] = pd.to_numeric(
        df_merged["nfl_player_id_1"], errors="coerce"
    )

    df_merged = df_merged.merge(
        df_helm,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_vis_1"),
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns to have _1 suffix
    # Tracking cols already have _t{i} suffix. We need to distinguish P1 vs P2.
    # The merge didn't add suffixes because keys matched.
    # We need to rename the feature columns coming from df_track.

    track_cols = [
        c for c in df_track.columns if c not in ["game_play", "step", "nfl_player_id"]
    ]
    vis_cols = [
        c for c in df_helm.columns if c not in ["game_play", "step", "nfl_player_id"]
    ]

    # Rename P1 columns
    rename_dict_1 = {c: f"{c}_1" for c in track_cols + vis_cols}
    df_merged = df_merged.rename(columns=rename_dict_1)

    # 4. Merge Player 2 (Handle Ground)
    # Identify Ground rows
    is_ground = df_merged["nfl_player_id_2"] == "G"

    # Create a temporary numeric ID column for merge, 'G' becomes NaN
    df_merged["nfl_player_id_2_num"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    # Merge Tracking P2
    df_merged = df_merged.merge(
        df_track,
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_2_track"),
    ).drop(columns=["nfl_player_id"])

    # Rename P2 columns
    rename_dict_2 = {c: f"{c}_2" for c in track_cols}
    df_merged = df_merged.rename(columns=rename_dict_2)

    # Merge Visuals P2
    df_merged = df_merged.merge(
        df_helm,
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_2_vis"),
    ).drop(columns=["nfl_player_id"])

    rename_dict_vis_2 = {c: f"{c}_2" for c in vis_cols}
    df_merged = df_merged.rename(columns=rename_dict_vis_2)

    # 5. Hybrid Ground Imputation
    # If P2 is Ground:
    # P2 Position = P1 Position (at t0)
    # P2 Speed/Accel = 0
    # P2 Visuals = 0

    # We need to identify the t0 columns for position imputation
    x_col_t0_1 = "x_position_t0_1"
    y_col_t0_1 = "y_position_t0_1"
    x_col_t0_2 = "x_position_t0_2"
    y_col_t0_2 = "y_position_t0_2"

    if is_ground.any():
        # Impute positions
        df_merged.loc[is_ground, x_col_t0_2] = df_merged.loc[is_ground, x_col_t0_1]
        df_merged.loc[is_ground, y_col_t0_2] = df_merged.loc[is_ground, y_col_t0_1]

        # For other time steps in the window, we should probably impute similarly
        # (Ground stays with player? or Ground is static at that location?
        # "Ground Position = Player Position" implies relative distance 0.
        # So we copy P1's trajectory to P2 for position, making distance 0 across window?
        # Or just set P2 to P1's current pos and 0 velocity?
        # "Ground Velocity = 0". This implies P2 should be static.
        # If P2 is static at P1's location, distance will grow as P1 moves.
        # But "Distance=0" suggestion implies we force distance to 0.
        # Let's force P2 position = P1 position for ALL timesteps to ensure Distance=0.
        for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            suffix = f"_t{offset:+d}" if offset != 0 else "_t0"
            df_merged.loc[is_ground, f"x_position{suffix}_2"] = df_merged.loc[
                is_ground, f"x_position{suffix}_1"
            ]
            df_merged.loc[is_ground, f"y_position{suffix}_2"] = df_merged.loc[
                is_ground, f"y_position{suffix}_1"
            ]

            # Zero out velocity/accel for Ground
            for feat in ["speed", "acceleration", "sa"]:
                col = f"{feat}{suffix}_2"
                if col in df_merged.columns:
                    df_merged.loc[is_ground, col] = 0.0

        # Zero out P2 visuals
        for col in vis_cols:
            df_merged.loc[is_ground, f"{col}_2"] = 0.0

    # Fill remaining NaNs (missing tracking for non-ground players) with 0
    # This handles cases where a player doesn't have tracking data
    feat_cols = [c for c in df_merged.columns if "_1" in c or "_2" in c]
    df_merged[feat_cols] = df_merged[feat_cols].fillna(0.0)

    # 6. Interaction Features (Distance, etc.)
    # Calculated at t0
    dx = df_merged[x_col_t0_1] - df_merged[x_col_t0_2]
    dy = df_merged[y_col_t0_1] - df_merged[y_col_t0_2]
    df_merged["distance"] = np.sqrt(dx**2 + dy**2)

    # Relative Speed
    s1 = df_merged["speed_t0_1"]
    s2 = df_merged["speed_t0_2"]
    df_merged["relative_speed"] = np.abs(
        s1 - s2
    )  # Simple scalar diff, vector diff would be better but this is baseline

    # Relative Angle
    # orientation is 0-360.
    o1 = df_merged["orientation_t0_1"]
    o2 = df_merged["orientation_t0_2"]
    df_merged["relative_angle"] = shortest_arc_distance(o1, o2)

    # 7. Scaling
    # Identify all continuous feature columns to scale
    # Exclude categorical encodings and metadata
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "nfl_player_id_2_num",
        "datetime",
        "contact",
        "position_enc_1",
        "team_enc_1",
        "position_enc_2",
        "team_enc_2",
    ]

    # Video paths are in metadata, exclude them
    exclude_cols += ["path_endzone", "path_sideline", "path_all29"]

    feature_cols = [c for c in df_merged.columns if c not in exclude_cols]
    # Filter out any non-numeric columns just in case
    feature_cols = (
        df_merged[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )

    scaler_path = _get_cache_path("scaler.joblib")

    if "train" in split_name:
        scaler = StandardScaler()
        # Fit on a sample if too large, but 3M rows is fine for StandardScaler
        scaler.fit(df_merged[feature_cols])
        joblib.dump(scaler, scaler_path)
    else:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            # Fallback (should not happen)
            scaler = StandardScaler()
            scaler.fit(df_merged[feature_cols])

    df_merged[feature_cols] = scaler.transform(df_merged[feature_cols])

    # Cast to float32
    df_merged[feature_cols] = df_merged[feature_cols].astype("float32")

    # Save final dataframe
    df_merged.to_parquet(final_cache_path)

    return df_merged


# =========================================================================
# Dataset Class
# =========================================================================


class ContactDataset(Dataset):
    def __init__(self, df, inference=False):
        self.inference = inference
        self.df = df

        # Organize feature groups

        # 1. Kinematic Features (Wide format)
        # Identify all columns related to kinematics (x, y, speed, etc.) for P1 and P2
        # We rely on the naming convention from preprocess_tracking_data
        # _t{i}_1 and _t{i}_2
        # Plus interaction features

        # We need a stable order.
        self.kin_cols = [
            c
            for c in df.columns
            if any(k in c for k in Config.KINEMATIC_FEATURES)
            and ("_1" in c or "_2" in c)
        ]
        self.kin_cols += Config.INTERACTION_FEATURES
        self.kin_cols = sorted(list(set(self.kin_cols)))  # Sort for determinism

        # 2. Visual Features
        self.vis_cols = [
            c
            for c in df.columns
            if any(v in c for v in Config.VISUAL_FEATURES) and ("_1" in c or "_2" in c)
        ]
        self.vis_cols = sorted(list(set(self.vis_cols)))

        # 3. Categorical Features
        self.cat_cols = ["position_enc_1", "team_enc_1", "position_enc_2", "team_enc_2"]

        # Convert to tensors
        self.X_kin = torch.tensor(df[self.kin_cols].values, dtype=torch.float32)
        self.X_vis = torch.tensor(df[self.vis_cols].values, dtype=torch.float32)
        self.X_cat = torch.tensor(df[self.cat_cols].fillna(0).values, dtype=torch.long)

        if not self.inference:
            self.y = torch.tensor(df["contact"].values, dtype=torch.float32)
        else:
            self.y = None
            self.contact_ids = df["contact_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.inference:
            return (
                self.X_kin[idx],
                self.X_vis[idx],
                self.X_cat[idx],
                self.contact_ids[idx],
            )
        else:
            return self.X_kin[idx], self.X_vis[idx], self.X_cat[idx], self.y[idx]

    def get_feature_dims(self):
        return {"kinematic": self.X_kin.shape[1], "visual": self.X_vis.shape[1]}


# =========================================================================
# Main Data Loading Interface
# =========================================================================


def get_dataset(split="train", load_cached_data=True):
    """
    Interface to get the ContactDataset for a specific split.
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        track_path = Config.TRAIN_TRACKING_PATH
        helm_path = Config.TRAIN_HELMETS_PATH
        inference = False
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        # Val uses train tracking/helmets as source
        track_path = Config.TRAIN_TRACKING_PATH
        helm_path = Config.TRAIN_HELMETS_PATH
        inference = False
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        track_path = Config.TEST_TRACKING_PATH
        helm_path = Config.TEST_HELMETS_PATH
        inference = True
    else:
        raise ValueError(f"Unknown split: {split}")

    df_features = create_features(
        meta_path, track_path, helm_path, split, load_cached_data
    )

    dataset = ContactDataset(df_features, inference=inference)
    return dataset
