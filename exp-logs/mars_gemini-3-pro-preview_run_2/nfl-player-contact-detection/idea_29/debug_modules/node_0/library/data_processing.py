import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class NFLDataset(Dataset):
    """
    PyTorch Dataset for SEA-RVN model.
    Returns:
        X_kin_cont: (N, D_kin) - Continuous kinematic features (wide window)
        X_kin_cat: (N, 4) - Categorical features [pos1, team1, pos2, team2]
        X_vis: (N, D_vis) - Visual features (wide window)
        y: (N,) - Target label (if available)
    """

    def __init__(self, X_kin_cont, X_kin_cat, X_vis, y=None):
        self.X_kin_cont = torch.FloatTensor(X_kin_cont)
        self.X_kin_cat = torch.LongTensor(X_kin_cat)
        self.X_vis = torch.FloatTensor(X_vis)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_kin_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return (
                self.X_kin_cont[idx],
                self.X_kin_cat[idx],
                self.X_vis[idx],
                self.y[idx],
            )
        return self.X_kin_cont[idx], self.X_kin_cat[idx], self.X_vis[idx]


def load_and_preprocess_tracking(path, relevant_gps=None):
    """
    Loads tracking data and generates wide-format windowed features.
    """
    df = pd.read_csv(path)

    # Filter to relevant game_plays if provided to save memory
    if relevant_gps is not None:
        df = df[df["game_play"].isin(relevant_gps)].copy()

    # Standardize angles (0-360)
    df["orientation"] = df["orientation"].fillna(0) % 360
    df["direction"] = df["direction"].fillna(0) % 360

    # Sort for windowing
    df = df.sort_values(["game_play", "nfl_player_id", "step"])

    # Features to window
    raw_feats = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]
    raw_feats = [c for c in raw_feats if c in df.columns]

    # Generate Windowed Features (Wide Format)
    # We use shift logic grouped by player
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    # Initialize output with keys
    wide_df = df[["game_play", "nfl_player_id", "step", "position", "team"]].copy()

    # Group object for shifting
    grp = df.groupby(["game_play", "nfl_player_id"])

    for col in raw_feats:
        for s in shifts:
            # shift(-s): s=1 (future t+1) needs shift(-1). s=-1 (past t-1) needs shift(1).
            wide_df[f"{col}_lag_{s}"] = grp[col].shift(-s)

    # Fill NAs at edges of play with 0 (or nearest? 0 is safer for standardization)
    wide_df = wide_df.fillna(0)

    return wide_df


def preprocess_helmets(path, relevant_gps=None):
    """
    Loads helmet data, applies Max-Pooling, and generates windowed features.
    """
    df = pd.read_csv(path)

    if relevant_gps is not None:
        df = df[df["game_play"].isin(relevant_gps)].copy()

    # Max-Pooling Selection Strategy: Select box with largest area per player/step
    df["area"] = df["width"] * df["height"]
    df = df.sort_values(
        ["game_play", "step", "nfl_player_id", "area"],
        ascending=[True, True, True, False],
    )
    df = df.drop_duplicates(subset=["game_play", "step", "nfl_player_id"], keep="first")

    # Sort for windowing
    df = df.sort_values(["game_play", "nfl_player_id", "step"])

    # Windowing
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)
    vis_feats = Config.VISUAL_FEATURES

    wide_df = df[["game_play", "nfl_player_id", "step"]].copy()
    grp = df.groupby(["game_play", "nfl_player_id"])

    for col in vis_feats:
        for s in shifts:
            wide_df[f"{col}_lag_{s}"] = grp[col].shift(-s)

    wide_df = wide_df.fillna(0)

    return wide_df


def get_derived_features(df):
    """
    Calculates pairwise derived features (Distance, Closing Speed, Relative Angle).
    Operates on the merged dataframe.
    """
    # Distance (at t=0)
    df["dx"] = df["x_position_lag_0_1"] - df["x_position_lag_0_2"]
    df["dy"] = df["y_position_lag_0_1"] - df["y_position_lag_0_2"]
    df["distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    # Closing Speed (Projected relative velocity)
    # v_closing = - (v_rel . r_rel) / |r_rel|
    def get_v(suffix):
        s = df[f"speed_lag_0_{suffix}"]
        d = np.deg2rad(df[f"direction_lag_0_{suffix}"])
        vx = s * np.sin(d)
        vy = s * np.cos(d)
        return vx, vy

    vx1, vy1 = get_v("1")
    vx2, vy2 = get_v("2")

    dvx = vx1 - vx2
    dvy = vy1 - vy2

    dist = df["distance"] + 1e-6
    # Positive closing speed = approaching
    df["closing_speed"] = -(dvx * df["dx"] + dvy * df["dy"]) / dist

    # Relative Angle
    d1 = df["direction_lag_0_1"]
    d2 = df["direction_lag_0_2"]
    diff = np.abs(d1 - d2)
    df["relative_angle"] = np.minimum(diff, 360 - diff)

    return df


def process_features(
    metadata_path,
    tracking_path,
    helmets_path,
    is_train=True,
    load_cached_data=True,
    cache_name=None,
):
    """
    Main processing pipeline: Merge -> Impute -> Derive -> Clamp.
    """
    # Cache handling
    if cache_name is None:
        mode = "train" if is_train else "test"
        cache_name = f"{mode}_processed.parquet"

    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 1. Load Metadata
    meta = pd.read_csv(metadata_path)
    if Config.DEBUG and is_train:
        meta = meta.sample(
            n=min(len(meta), Config.DEBUG_SAMPLES), random_state=Config.SEED
        ).copy()

    relevant_gps = meta["game_play"].unique()

    # 2. Preprocess Inputs (Entity-First)
    track_wide = load_and_preprocess_tracking(tracking_path, relevant_gps)
    vis_wide = preprocess_helmets(helmets_path, relevant_gps)

    # 3. Merge P1
    # Ensure ID types
    meta["nfl_player_id_1"] = pd.to_numeric(meta["nfl_player_id_1"], errors="coerce")

    # Merge Tracking P1
    df = meta.merge(
        track_wide,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )
    # Rename P1 columns
    p1_track_cols = [
        c for c in track_wide.columns if c not in ["game_play", "nfl_player_id", "step"]
    ]
    df = df.rename(columns={c: f"{c}_1" for c in p1_track_cols})

    # Merge Visual P1
    df = df.merge(
        vis_wide,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )
    p1_vis_cols = [
        c for c in vis_wide.columns if c not in ["game_play", "nfl_player_id", "step"]
    ]
    df = df.rename(columns={c: f"{c}_1" for c in p1_vis_cols})

    # 4. Merge P2
    meta["is_ground"] = (meta["nfl_player_id_2"] == "G").astype(int)
    meta["nfl_player_id_2_num"] = pd.to_numeric(
        meta["nfl_player_id_2"], errors="coerce"
    )

    # We update df with these new columns from meta (merge dropped them? No, we merged onto meta)
    df["is_ground"] = meta["is_ground"]
    df["nfl_player_id_2_num"] = meta["nfl_player_id_2_num"]

    # Merge Tracking P2
    df = df.merge(
        track_wide,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )
    df = df.rename(columns={c: f"{c}_2" for c in p1_track_cols})

    # Merge Visual P2
    df = df.merge(
        vis_wide,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )
    df = df.rename(columns={c: f"{c}_2" for c in p1_vis_cols})

    # 5. Imputation (Ground & Missing)
    ground_mask = df["is_ground"] == 1
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    for s in shifts:
        # Ground Position = P1 Position
        df.loc[ground_mask, f"x_position_lag_{s}_2"] = df.loc[
            ground_mask, f"x_position_lag_{s}_1"
        ]
        df.loc[ground_mask, f"y_position_lag_{s}_2"] = df.loc[
            ground_mask, f"y_position_lag_{s}_1"
        ]

        # Ground Dynamics = 0
        for attr in ["speed", "acceleration", "sa", "orientation", "direction"]:
            df.loc[ground_mask, f"{attr}_lag_{s}_2"] = 0

        # Ground Visuals = 0
        for attr in Config.VISUAL_FEATURES:
            df.loc[ground_mask, f"{attr}_lag_{s}_2"] = 0

    # Fill remaining missing data (e.g. lost tracking) with 0
    df = df.fillna(0)

    # 6. Derived Features
    df = get_derived_features(df)

    # 7. Clamping (Numerical Stability)
    # Collect all continuous columns
    cont_cols = []
    track_base = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]
    for col in track_base:
        for s in shifts:
            cont_cols.append(f"{col}_lag_{s}_1")
            cont_cols.append(f"{col}_lag_{s}_2")
    cont_cols.extend(["distance", "closing_speed", "relative_angle"])

    for col in cont_cols:
        if col in df.columns:
            df[col] = df[col].clip(Config.CLAMP_MIN, Config.CLAMP_MAX)

    # Save to cache
    df.to_parquet(cache_path)

    return df


def get_datasets(load_cached_data=True):
    """
    Generates Train and Validation Datasets.
    Fits Scalers/Encoders on Train, applies to Val.
    """
    # Process Data
    # Note: Config.TRAIN_META is the 80% split, Config.VAL_META is the 20% split.
    df_train = process_features(
        Config.TRAIN_META,
        Config.TRAIN_TRACKING,
        Config.TRAIN_HELMETS,
        is_train=True,
        load_cached_data=load_cached_data,
        cache_name="train_features.parquet",
    )

    df_val = process_features(
        Config.VAL_META,
        Config.TRAIN_TRACKING,
        Config.TRAIN_HELMETS,
        is_train=True,
        load_cached_data=load_cached_data,
        cache_name="val_features.parquet",
    )

    # Define Column Groups
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    # Kinematic Continuous
    kin_cont_cols = []
    track_base = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]
    for col in track_base:
        for s in shifts:
            kin_cont_cols.append(f"{col}_lag_{s}_1")
            kin_cont_cols.append(f"{col}_lag_{s}_2")
    kin_cont_cols.extend(["distance", "closing_speed", "relative_angle"])

    # Visual Continuous
    vis_cols = []
    for col in Config.VISUAL_FEATURES:
        for s in shifts:
            vis_cols.append(f"{col}_lag_{s}_1")
            vis_cols.append(f"{col}_lag_{s}_2")

    # Categorical
    cat_cols = ["position_1", "team_1", "position_2", "team_2"]

    # Scaling & Encoding
    scaler_cont = StandardScaler()
    X_train_cont = scaler_cont.fit_transform(df_train[kin_cont_cols])
    X_val_cont = scaler_cont.transform(df_val[kin_cont_cols])

    scaler_vis = StandardScaler()
    X_train_vis = scaler_vis.fit_transform(df_train[vis_cols])
    X_val_vis = scaler_vis.transform(df_val[vis_cols])

    encoders = {}
    X_train_cat_list = []
    X_val_cat_list = []

    for col in cat_cols:
        le = LabelEncoder()
        # Fit on train, handle unknowns in val/test via mapping
        # Convert to string to be safe
        train_vals = df_train[col].astype(str)
        le.fit(train_vals)
        encoders[col] = le

        X_train_cat_list.append(le.transform(train_vals))

        # Safe transform for val
        val_vals = df_val[col].astype(str)
        known = set(le.classes_)
        # Map unknown to first class (usually index 0)
        val_vals = val_vals.apply(lambda x: x if x in known else le.classes_[0])
        X_val_cat_list.append(le.transform(val_vals))

    X_train_cat = np.stack(X_train_cat_list, axis=1)
    X_val_cat = np.stack(X_val_cat_list, axis=1)

    # Targets
    y_train = df_train["contact"].values
    y_val = df_val["contact"].values

    # Save Artifacts
    joblib.dump(scaler_cont, os.path.join(Config.WORKING_DIR, "scaler_cont.joblib"))
    joblib.dump(scaler_vis, os.path.join(Config.WORKING_DIR, "scaler_vis.joblib"))
    joblib.dump(encoders, os.path.join(Config.WORKING_DIR, "encoders.joblib"))

    # Create Datasets
    train_ds = NFLDataset(X_train_cont, X_train_cat, X_train_vis, y_train)
    val_ds = NFLDataset(X_val_cont, X_val_cat, X_val_vis, y_val)

    return train_ds, val_ds


def get_test_dataset(load_cached_data=True):
    """
    Generates Test Dataset using pre-fitted scalers.
    Returns Dataset and contact_ids.
    """
    df_test = process_features(
        Config.TEST_META,
        Config.TEST_TRACKING,
        Config.TEST_HELMETS,
        is_train=False,
        load_cached_data=load_cached_data,
        cache_name="test_features.parquet",
    )

    # Load Artifacts
    scaler_cont = joblib.load(os.path.join(Config.WORKING_DIR, "scaler_cont.joblib"))
    scaler_vis = joblib.load(os.path.join(Config.WORKING_DIR, "scaler_vis.joblib"))
    encoders = joblib.load(os.path.join(Config.WORKING_DIR, "encoders.joblib"))

    # Feature Columns (Must match train)
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    kin_cont_cols = []
    track_base = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]
    for col in track_base:
        for s in shifts:
            kin_cont_cols.append(f"{col}_lag_{s}_1")
            kin_cont_cols.append(f"{col}_lag_{s}_2")
    kin_cont_cols.extend(["distance", "closing_speed", "relative_angle"])

    vis_cols = []
    for col in Config.VISUAL_FEATURES:
        for s in shifts:
            vis_cols.append(f"{col}_lag_{s}_1")
            vis_cols.append(f"{col}_lag_{s}_2")

    cat_cols = ["position_1", "team_1", "position_2", "team_2"]

    # Transform
    X_cont = scaler_cont.transform(df_test[kin_cont_cols])
    X_vis = scaler_vis.transform(df_test[vis_cols])

    X_cat_list = []
    for col in cat_cols:
        le = encoders[col]
        vals = df_test[col].astype(str)
        known = set(le.classes_)
        vals = vals.apply(lambda x: x if x in known else le.classes_[0])
        X_cat_list.append(le.transform(vals))

    X_cat = np.stack(X_cat_list, axis=1)

    ds = NFLDataset(X_cont, X_cat, X_vis, y=None)
    return ds, df_test["contact_id"]
