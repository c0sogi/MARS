import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config

# Set seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class ContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    """

    def __init__(self, features, labels=None):
        """
        Args:
            features (np.ndarray): Feature matrix (N, D)
            labels (np.ndarray, optional): Label vector (N,). Defaults to None.
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


def preprocess_tracking(df_tracking):
    """
    Generates temporal lag features for tracking data.
    Returns a dataframe with columns for each feature at each timestep in the window.
    """
    # Sort to ensure correct shifting
    df_tracking = df_tracking.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    feature_cols = Config.PLAYER_LAG_FEATURES
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    # We will construct a list of series/dfs to concatenate
    dfs_to_concat = [df_tracking[["game_play", "nfl_player_id", "step"]]]

    # Group by object to shift within boundaries
    grouped = df_tracking.groupby(["game_play", "nfl_player_id"])

    for col in feature_cols:
        for s in shifts:
            # s is the time offset relative to t=0.
            # If s = -1 (t-1), we want the PREVIOUS row, so shift(1).
            # If s = 1 (t+1), we want the NEXT row, so shift(-1).
            shift_amount = -s

            col_name = f"{col}_t{s}"

            # Generate shifted series
            series = grouped[col].shift(shift_amount)
            series.name = col_name
            dfs_to_concat.append(series)

    # Concatenate all features
    df_processed = pd.concat(dfs_to_concat, axis=1)

    # Fill NaNs at the edges of plays with 0 (standard padding for wide windows)
    df_processed = df_processed.fillna(0)

    return df_processed


def create_interaction_features(df):
    """
    Calculates explicit relative physics and interaction features for all timesteps in the window.
    """
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    for s in shifts:
        suffix = f"_t{s}"

        # Extract coordinates and velocities
        x1 = df[f"x_position_1{suffix}"]
        y1 = df[f"y_position_1{suffix}"]
        x2 = df[f"x_position_2{suffix}"]
        y2 = df[f"y_position_2{suffix}"]

        # Decompose speed into vector components using direction (0 deg = North/Y, 90 deg = East/X)
        # Note: We use sin/cos consistently. Even if the frame is rotated, relative magnitude is invariant.
        dir1_rad = np.deg2rad(df[f"direction_1{suffix}"])
        vx1 = df[f"speed_1{suffix}"] * np.sin(dir1_rad)
        vy1 = df[f"speed_1{suffix}"] * np.cos(dir1_rad)

        dir2_rad = np.deg2rad(df[f"direction_2{suffix}"])
        vx2 = df[f"speed_2{suffix}"] * np.sin(dir2_rad)
        vy2 = df[f"speed_2{suffix}"] * np.cos(dir2_rad)

        # Distance
        dx = x1 - x2
        dy = y1 - y2
        dist = np.sqrt(dx**2 + dy**2)

        df[f"distance{suffix}"] = dist
        df[f"log_distance{suffix}"] = np.log1p(dist)

        # Relative Speed (Magnitude of velocity difference)
        dvx = vx1 - vx2
        dvy = vy1 - vy2
        rel_speed = np.sqrt(dvx**2 + dvy**2)
        df[f"relative_speed{suffix}"] = rel_speed

        # Closing Speed: Projection of relative velocity onto relative position vector
        # closing = -(v_rel . p_rel) / |p_rel|
        dot_prod = dvx * dx + dvy * dy
        dist_clamped = np.maximum(dist, 1e-6)  # Avoid div by zero
        df[f"closing_speed{suffix}"] = -(dot_prod / dist_clamped)

        # Relative Acceleration (Scalar difference proxy)
        a1 = df[f"acceleration_1{suffix}"]
        a2 = df[f"acceleration_2{suffix}"]
        df[f"relative_acceleration{suffix}"] = np.abs(a1 - a2)

    return df


def merge_and_impute(df_labels, df_tracking_processed):
    """
    Merges labels with processed tracking data and handles Hybrid Ground Imputation.
    """
    # Identify feature columns in processed tracking
    feature_cols = [c for c in df_tracking_processed.columns if "_t" in c]

    # --- Merge Player 1 ---
    df = df_labels.merge(
        df_tracking_processed,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )
    # Rename columns to _1
    rename_dict_1 = {
        c: f"{c.split('_t')[0]}_1_t{c.split('_t')[1]}" for c in feature_cols
    }
    df = df.rename(columns=rename_dict_1)
    df = df.drop(columns=["nfl_player_id"])

    # --- Merge Player 2 ---
    # Convert nfl_player_id_2 to numeric for join (forces 'G' to NaN)
    df["nfl_player_id_2_join"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

    df = df.merge(
        df_tracking_processed,
        left_on=["game_play", "nfl_player_id_2_join", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_2"),
    )

    # Rename columns to _2
    rename_dict_2 = {
        c: f"{c.split('_t')[0]}_2_t{c.split('_t')[1]}" for c in feature_cols
    }
    df = df.rename(columns=rename_dict_2)
    df = df.drop(columns=["nfl_player_id", "nfl_player_id_2_join"])

    # --- Hybrid Ground Imputation ---
    is_ground = df["nfl_player_id_2"] == "G"
    df["is_ground"] = is_ground.astype(int)

    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    for s in shifts:
        suffix = f"_t{s}"

        # Position Imputation: Ground is at Player's location (dist=0)
        for coord in ["x_position", "y_position"]:
            col_1 = f"{coord}_1{suffix}"
            col_2 = f"{coord}_2{suffix}"
            # Where is_ground is True, overwrite col_2 with col_1
            df[col_2] = np.where(is_ground, df[col_1], df[col_2])

        # Kinematic Imputation: Ground has 0 velocity/acceleration
        for kin in ["speed", "acceleration", "sa", "orientation", "direction"]:
            col_2 = f"{kin}_2{suffix}"
            df[col_2] = np.where(is_ground, 0, df[col_2])

    # Fill remaining NaNs (missing tracking for existing players) with 0
    all_feature_cols = [c for c in df.columns if "_t" in c]
    df[all_feature_cols] = df[all_feature_cols].fillna(0)

    return df


def get_data_loaders(load_cached_data=True):
    """
    Main entry point. Handles caching, processing, and DataLoader creation.

    Returns:
        train_loader, val_loader, center_indices, scaler
    """
    cache_dir = Config.WORKING_DIR
    cache_train_path = os.path.join(cache_dir, "train_features.parquet")
    cache_val_path = os.path.join(cache_dir, "val_features.parquet")
    cache_scaler_path = os.path.join(cache_dir, "scaler.joblib")
    cache_meta_path = os.path.join(cache_dir, "feature_meta.joblib")

    if (
        load_cached_data
        and os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
    ):
        print("Loading cached features from disk...")
        X_train_df = pd.read_parquet(cache_train_path)
        X_val_df = pd.read_parquet(cache_val_path)
        scaler = joblib.load(cache_scaler_path)
        meta = joblib.load(cache_meta_path)
        center_indices = meta["center_indices"]
        feature_cols = meta["feature_cols"]

    else:
        print("Processing data from scratch...")

        # Load Metadata
        df_train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
        df_val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))

        if Config.DEBUG:
            print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows...")
            df_train_meta = df_train_meta.sample(
                min(len(df_train_meta), Config.DEBUG_SAMPLE_SIZE),
                random_state=Config.SEED,
            )
            df_val_meta = df_val_meta.sample(
                min(len(df_val_meta), int(Config.DEBUG_SAMPLE_SIZE / 4)),
                random_state=Config.SEED,
            )

        # Load Tracking
        needed_gps = set(df_train_meta["game_play"]).union(
            set(df_val_meta["game_play"])
        )
        print("Loading tracking data...")
        df_tracking = pd.read_csv(
            os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
        )
        df_tracking = df_tracking[df_tracking["game_play"].isin(needed_gps)].copy()

        # Preprocess Tracking (Lags)
        print("Generating lag features...")
        df_tracking_proc = preprocess_tracking(df_tracking)

        # Merge and Impute
        print("Merging and imputing data...")
        df_train_merged = merge_and_impute(df_train_meta, df_tracking_proc)
        df_val_merged = merge_and_impute(df_val_meta, df_tracking_proc)

        # Interaction Features
        print("Calculating interaction features...")
        df_train_merged = create_interaction_features(df_train_merged)
        df_val_merged = create_interaction_features(df_val_merged)

        # Feature Selection
        feature_cols = [c for c in df_train_merged.columns if "_t" in c]
        feature_cols.append("is_ground")
        feature_cols = sorted(feature_cols)

        X_train = df_train_merged[feature_cols].values.astype(np.float32)
        y_train = df_train_merged["contact"].values.astype(np.float32)

        X_val = df_val_merged[feature_cols].values.astype(np.float32)
        y_val = df_val_merged["contact"].values.astype(np.float32)

        # Scaling
        print("Fitting scaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Identify Center Indices for Skip Connection
        center_indices = []
        for idx, col in enumerate(feature_cols):
            # Check if col is a t0 feature and in the allowed center list
            if col.endswith("_t0"):
                base_name = col[:-3]  # remove _t0
                if base_name in Config.CENTER_FEATURE_NAMES:
                    center_indices.append(idx)
            elif col == "is_ground" and "is_ground" in Config.CENTER_FEATURE_NAMES:
                center_indices.append(idx)

        print(f"Identified {len(center_indices)} center features.")

        # Save to Cache
        print("Saving to cache...")
        train_save = pd.DataFrame(X_train, columns=feature_cols)
        train_save["contact"] = y_train

        val_save = pd.DataFrame(X_val, columns=feature_cols)
        val_save["contact"] = y_val

        train_save.to_parquet(cache_train_path)
        val_save.to_parquet(cache_val_path)
        joblib.dump(scaler, cache_scaler_path)
        joblib.dump(
            {"center_indices": center_indices, "feature_cols": feature_cols},
            cache_meta_path,
        )

        X_train_df = train_save
        X_val_df = val_save

    # Create DataLoaders
    feat_cols = [c for c in X_train_df.columns if c != "contact"]

    train_dataset = ContactDataset(
        X_train_df[feat_cols].values.astype(np.float32),
        X_train_df["contact"].values.astype(np.float32),
    )

    val_dataset = ContactDataset(
        X_val_df[feat_cols].values.astype(np.float32),
        X_val_df["contact"].values.astype(np.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, center_indices, scaler


def process_test_data(scaler):
    """
    Processes test data for inference using the fitted scaler.
    Returns test_loader and contact_ids.
    """
    df_test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    df_tracking = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "test_player_tracking.csv")
    )

    # Process
    df_tracking_proc = preprocess_tracking(df_tracking)
    df_test_merged = merge_and_impute(df_test_meta, df_tracking_proc)
    df_test_merged = create_interaction_features(df_test_merged)

    # Load feature columns from cache metadata to ensure alignment
    cache_meta_path = os.path.join(Config.WORKING_DIR, "feature_meta.joblib")
    if os.path.exists(cache_meta_path):
        meta = joblib.load(cache_meta_path)
        feature_cols = meta["feature_cols"]
    else:
        # Fallback
        feature_cols = [c for c in df_test_merged.columns if "_t" in c]
        feature_cols.append("is_ground")
        feature_cols = sorted(feature_cols)

    X_test = df_test_merged[feature_cols].values.astype(np.float32)
    X_test = scaler.transform(X_test)

    test_dataset = ContactDataset(X_test, labels=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    return test_loader, df_test_merged["contact_id"].values
