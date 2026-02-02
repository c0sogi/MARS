import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything

# Set global seed for reproducibility
seed_everything(Config.SEED)


def load_and_merge_tracking(metadata_path, tracking_path):
    """
    Loads metadata and tracking data, merges them based on game_play, step, and player IDs.
    Handles both Player-Player and Player-Ground interactions efficiently.
    """
    # Load Metadata
    df_meta = pd.read_csv(metadata_path)

    # Load Tracking Data
    # Read only necessary columns to optimize memory usage
    track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
    ]
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Tracking file not found: {tracking_path}")

    df_track = pd.read_csv(tracking_path, usecols=track_cols)

    # Ensure consistent types for merging
    df_meta["game_play"] = df_meta["game_play"].astype(str)
    df_track["game_play"] = df_track["game_play"].astype(str)

    # Ensure numeric merge keys are integers to prevent mismatches
    df_track.dropna(subset=["step", "nfl_player_id"], inplace=True)
    df_track["step"] = df_track["step"].astype(int)
    df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(int)

    df_meta["step"] = df_meta["step"].astype(int)
    df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

    # --- Merge Player 1 ---
    # Player 1 is always an integer ID
    df_merged = pd.merge(
        df_meta,
        df_track,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename Player 1 columns
    rename_p1 = {
        "x_position": "x_p1",
        "y_position": "y_p1",
        "speed": "speed_p1",
        "acceleration": "acc_p1",
        "orientation": "orientation_p1",
    }
    df_merged.rename(columns=rename_p1, inplace=True)
    df_merged.drop(columns=["nfl_player_id"], inplace=True)

    # --- Merge Player 2 ---
    # Player 2 can be an integer ID or 'G' (Ground)

    # Split dataset into Player-Ground and Player-Player interactions
    mask_ground = df_merged["nfl_player_id_2"] == "G"
    df_ground = df_merged[mask_ground].copy()
    df_players = df_merged[~mask_ground].copy()

    # 1. Handle Player-Player interactions
    if not df_players.empty:
        df_players["nfl_player_id_2"] = df_players["nfl_player_id_2"].astype(int)
        df_players = pd.merge(
            df_players,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        rename_p2 = {
            "x_position": "x_p2",
            "y_position": "y_p2",
            "speed": "speed_p2",
            "acceleration": "acc_p2",
            "orientation": "orientation_p2",
        }
        df_players.rename(columns=rename_p2, inplace=True)
        df_players.drop(columns=["nfl_player_id"], inplace=True)

    # 2. Handle Player-Ground interactions
    # Ground has no tracking data; initialize kinematic columns with 0.0
    if not df_ground.empty:
        for col in ["x_p2", "y_p2", "speed_p2", "acc_p2", "orientation_p2"]:
            df_ground[col] = 0.0

    # Recombine datasets
    df_final = pd.concat([df_players, df_ground], axis=0).sort_index()

    # Drop rows where tracking data is missing for Player 1
    df_final.dropna(subset=["x_p1"], inplace=True)

    # Drop rows where tracking data is missing for Player 2 (only applies to Player-Player)
    # Ground interactions have filled 0.0, so they won't be dropped here
    df_final.dropna(subset=["x_p2"], inplace=True)

    return df_final


def engineer_features(df):
    """
    Calculates kinematic features required for the model based on Config.FEATURES.
    """
    # Binary flag for ground contact
    df["is_ground"] = (df["nfl_player_id_2"] == "G").astype(int)

    # Euclidean Distance
    # For ground interactions, we set distance to 0.0 to distinguish from player separation
    d_x = df["x_p1"] - df["x_p2"]
    d_y = df["y_p1"] - df["y_p2"]
    dist = np.sqrt(d_x**2 + d_y**2)
    df["distance"] = np.where(df["is_ground"] == 1, 0.0, dist)

    # Kinematic Differences
    df["speed_diff"] = np.abs(df["speed_p1"] - df["speed_p2"])
    df["acc_diff"] = np.abs(df["acc_p1"] - df["acc_p2"])

    # Orientation Difference (Simple absolute difference, wrapped to 180 degrees)
    raw_diff = np.abs(df["orientation_p1"] - df["orientation_p2"])
    df["orientation_diff"] = np.minimum(raw_diff, 360 - raw_diff)

    # Ensure all configured features exist in the dataframe
    for feat in Config.FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    return df


def preprocess_data(df, mode="train", scaler=None):
    """
    Applies undersampling (for training data) and standard scaling.

    Args:
        df (pd.DataFrame): The dataframe with engineered features.
        mode (str): 'train', 'val', or 'test'.
        scaler (StandardScaler): Scaler object (required for val/test).

    Returns:
        tuple: (X_scaled, y, ids, scaler)
    """
    # 1. Undersampling (Train only)
    if mode == "train":
        # Separate positive (contact) and negative (no-contact) samples
        pos = df[df["contact"] == 1]
        neg = df[df["contact"] == 0]

        # Calculate number of negative samples to keep
        n_pos = len(pos)
        n_neg = int(n_pos * Config.UNDERSAMPLE_RATIO)

        # Sample negatives
        if len(neg) > n_neg:
            neg = neg.sample(n=n_neg, random_state=Config.SEED)

        # Combine and shuffle
        df_processed = (
            pd.concat([pos, neg])
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )
    else:
        df_processed = df.copy()

    # 2. Scaling
    features = Config.FEATURES
    # Convert to float32 for memory efficiency
    X = df_processed[features].values.astype(np.float32)

    if mode == "train":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for validation/test modes.")
        X_scaled = scaler.transform(X)

    # 3. Extract Targets and IDs
    ids = df_processed["contact_id"].values
    y = None
    if "contact" in df_processed.columns:
        y = df_processed["contact"].values.astype(np.float32)

    return X_scaled, y, ids, scaler


def get_processed_dataset(
    mode, metadata_path, tracking_path, scaler=None, load_cached_data=True
):
    """
    Orchestrates data loading, processing, and caching.

    Args:
        mode (str): 'train', 'val', or 'test'.
        metadata_path (str): Path to metadata CSV.
        tracking_path (str): Path to tracking CSV.
        scaler (StandardScaler): Scaler to use (required for val/test).
        load_cached_data (bool): Whether to use cached data if available.

    Returns:
        tuple: (X, y, ids, scaler)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.parquet")
    scaler_mean_path = os.path.join(Config.CACHE_DIR, "scaler_mean.npy")
    scaler_scale_path = os.path.join(Config.CACHE_DIR, "scaler_scale.npy")

    # --- Try Loading from Cache ---
    if load_cached_data:
        if os.path.exists(cache_path):
            # For train mode, ensure scaler files also exist
            scaler_available = True
            if mode == "train":
                if not (
                    os.path.exists(scaler_mean_path)
                    and os.path.exists(scaler_scale_path)
                ):
                    scaler_available = False

            if scaler_available:
                print(f"Loading {mode} data from cache: {cache_path}")
                df_cached = pd.read_parquet(cache_path)

                features = Config.FEATURES
                X = df_cached[features].values.astype(np.float32)
                ids = df_cached["contact_id"].values
                y = None
                if "contact" in df_cached.columns:
                    y = df_cached["contact"].values.astype(np.float32)

                # Reconstruct scaler for train mode
                if mode == "train":
                    scaler = StandardScaler()
                    scaler.mean_ = np.load(scaler_mean_path)
                    scaler.scale_ = np.load(scaler_scale_path)
                    scaler.var_ = scaler.scale_**2

                return X, y, ids, scaler

    # --- Compute from Scratch ---
    print(f"Processing {mode} data from scratch...")

    # 1. Load & Merge
    df_merged = load_and_merge_tracking(metadata_path, tracking_path)

    # 2. Engineer Features
    df_features = engineer_features(df_merged)

    # 3. Preprocess (Scale + Undersample)
    X, y, ids, scaler = preprocess_data(df_features, mode=mode, scaler=scaler)

    # --- Save to Cache ---
    # Save Data
    df_save = pd.DataFrame(X, columns=Config.FEATURES)
    df_save["contact_id"] = ids
    if y is not None:
        df_save["contact"] = y

    df_save.to_parquet(cache_path, index=False)
    print(f"Saved {mode} data to {cache_path}")

    # Save Scaler Parameters (Train only)
    if mode == "train":
        np.save(scaler_mean_path, scaler.mean_)
        np.save(scaler_scale_path, scaler.scale_)
        print(f"Saved scaler parameters to {Config.CACHE_DIR}")

    return X, y, ids, scaler
