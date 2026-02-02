import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import gc
import hashlib
import json

# Import configuration
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    WORKING_DIR,
    SEED,
    WINDOW_SIZE,
    TRACKING_FEATURES,
    UNDERSAMPLE_RATIO,
    SCALER_SAVE_PATH,
    setup_reproducibility,
)

# Set seeds for reproducibility
setup_reproducibility(SEED)


def load_raw_data(split):
    """
    Loads the metadata and appropriate tracking data for the given split.
    """
    print(f"Loading raw data for split: {split}")

    # Load Metadata
    if split == "train":
        df_meta = pd.read_csv(TRAIN_META_PATH)
        tracking_path = TRAIN_TRACKING_PATH
    elif split == "validation":
        df_meta = pd.read_csv(VAL_META_PATH)
        # Validation comes from train source, so use train tracking
        tracking_path = TRAIN_TRACKING_PATH
    elif split == "test":
        df_meta = pd.read_csv(TEST_META_PATH)
        tracking_path = TEST_TRACKING_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load Tracking Data
    # Optimization: Only load columns we need plus join keys
    tracking_cols = ["game_play", "step", "nfl_player_id"] + TRACKING_FEATURES
    df_tracking = pd.read_csv(tracking_path, usecols=tracking_cols)

    # Standardize IDs to string for merging
    df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
    df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
    df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

    # Filter tracking data to only include games present in metadata to save memory
    relevant_games = df_meta["game_play"].unique()
    df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_games)].copy()

    # Feature Engineering on Tracking Data (Cyclical Encoding)
    # Convert angular features to sin/cos components
    for col in ["direction", "orientation"]:
        if col in df_tracking.columns:
            rads = np.deg2rad(df_tracking[col])
            df_tracking[f"{col}_sin"] = np.sin(rads)
            df_tracking[f"{col}_cos"] = np.cos(rads)
            df_tracking.drop(columns=[col], inplace=True)

    return df_meta, df_tracking


def engineer_features(df_meta, df_tracking):
    """
    Constructs time-windowed features by merging tracking data onto the metadata labels.
    """
    print("Engineering features with temporal windows...")

    # Identify feature columns dynamically from the dataframe
    # This ensures new features (like sin/cos) are automatically included
    tracking_cols = [
        c
        for c in df_tracking.columns
        if c not in ["game_play", "step", "nfl_player_id"]
    ]

    # Base DataFrame
    df_features = df_meta[
        ["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
    ].copy()

    # We will collect feature names to return later
    feature_names = []

    # Iterate through the window
    # e.g., if WINDOW_SIZE=2, offsets are [-2, -1, 0, 1, 2]
    offsets = range(-WINDOW_SIZE, WINDOW_SIZE + 1)

    for offset in offsets:
        suffix = (
            f"_t{offset}" if offset < 0 else f"_t+{offset}" if offset > 0 else "_t0"
        )

        # Create a temporary step column for joining
        # We want the tracking data at (current_step + offset)
        df_features[f"join_step_{offset}"] = df_features["step"] + offset

        # --- Merge Player 1 ---
        # Join keys: game_play, join_step, nfl_player_id
        p1_suffix = f"_p1{suffix}"
        df_features = pd.merge(
            df_features,
            df_tracking.add_suffix(p1_suffix),
            left_on=["game_play", f"join_step_{offset}", "nfl_player_id_1"],
            right_on=[
                f"game_play{p1_suffix}",
                f"step{p1_suffix}",
                f"nfl_player_id{p1_suffix}",
            ],
            how="left",
        )

        # --- Merge Player 2 ---
        p2_suffix = f"_p2{suffix}"
        df_features = pd.merge(
            df_features,
            df_tracking.add_suffix(p2_suffix),
            left_on=["game_play", f"join_step_{offset}", "nfl_player_id_2"],
            right_on=[
                f"game_play{p2_suffix}",
                f"step{p2_suffix}",
                f"nfl_player_id{p2_suffix}",
            ],
            how="left",
        )

        # --- Handle Ground Contact & Missing Data ---
        # If nfl_player_id_2 is 'G', the merge above resulted in NaNs for p2 features.

        # Fill NaNs for P2 features with 0 (covers Ground and missing tracking)
        p2_cols = [f"{col}{p2_suffix}" for col in tracking_cols]
        df_features[p2_cols] = df_features[p2_cols].fillna(0)

        # Fill NaNs for P1 features with 0 (missing tracking)
        p1_cols = [f"{col}{p1_suffix}" for col in tracking_cols]
        df_features[p1_cols] = df_features[p1_cols].fillna(0)

        # --- Derived Features ---
        # Distance
        x1 = df_features[f"x_position{p1_suffix}"]
        y1 = df_features[f"y_position{p1_suffix}"]
        x2 = df_features[f"x_position{p2_suffix}"]
        y2 = df_features[f"y_position{p2_suffix}"]

        dist_col = f"distance{suffix}"
        df_features[dist_col] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Add to feature list
        feature_names.extend(p1_cols)
        feature_names.extend(p2_cols)
        feature_names.append(dist_col)

        # Cleanup merge columns to save memory
        drop_cols = [
            f"game_play{p1_suffix}",
            f"step{p1_suffix}",
            f"nfl_player_id{p1_suffix}",
            f"game_play{p2_suffix}",
            f"step{p2_suffix}",
            f"nfl_player_id{p2_suffix}",
            f"join_step_{offset}",
        ]
        df_features.drop(columns=drop_cols, inplace=True, errors="ignore")

    # --- Finalize Features ---
    # Add is_ground flag
    df_features["is_ground"] = (df_features["nfl_player_id_2"] == "G").astype(int)
    feature_names.append("is_ground")

    # Fix distance for Ground contacts
    # If is_ground == 1, set all distance columns to 0
    dist_cols = [f for f in feature_names if "distance" in f]
    df_features.loc[df_features["is_ground"] == 1, dist_cols] = 0

    # Extract X and y
    X = df_features[feature_names]

    # Get target if available (train/val), else None (test)
    if "contact" in df_meta.columns:
        y = df_meta["contact"].values
    else:
        y = np.zeros(len(df_features))  # Placeholder for test

    ids = df_features["contact_id"].values

    return X, y, ids, feature_names


def get_data(split="train", load_cached_data=True):
    """
    Main entry point to get processed data (X, y, ids).
    Handles caching, undersampling (for train), and scaling.
    """
    # Generate hash based on configuration to ensure cache validity
    # Cite solution_lesson_node_00004
    config_state = {
        "window_size": WINDOW_SIZE,
        "feature_version": "v2_cyclic",
        "undersample": UNDERSAMPLE_RATIO,
    }
    config_str = json.dumps(config_state, sort_keys=True).encode()
    config_hash = hashlib.md5(config_str).hexdigest()[:8]

    cache_prefix = os.path.join(WORKING_DIR, f"{split}_{config_hash}")
    cache_X = f"{cache_prefix}_X.parquet"
    cache_y = f"{cache_prefix}_y.npy"
    cache_ids = f"{cache_prefix}_ids.npy"

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"Loading cached data from {WORKING_DIR}...")
        X = pd.read_parquet(cache_X)
        y = np.load(cache_y)
        ids = np.load(cache_ids, allow_pickle=True)
        return X, y, ids

    # 2. Process from Scratch
    df_meta, df_tracking = load_raw_data(split)
    X, y, ids, feature_names = engineer_features(df_meta, df_tracking)

    # Clean up raw data
    del df_meta, df_tracking
    gc.collect()

    # 3. Split-Specific Processing
    if split == "train":
        # --- Scaling ---
        print("Fitting scaler on training data...")
        scaler = StandardScaler()
        # Fit on all data before undersampling to capture true distribution statistics
        X[feature_names] = scaler.fit_transform(X[feature_names])
        dump(scaler, SCALER_SAVE_PATH)
        print(f"Scaler saved to {SCALER_SAVE_PATH}")

        # --- Undersampling ---
        print(f"Undersampling majority class with ratio {UNDERSAMPLE_RATIO}...")
        pos_mask = y == 1
        neg_mask = y == 0

        pos_ids = np.where(pos_mask)[0]
        neg_ids = np.where(neg_mask)[0]

        n_pos = len(pos_ids)
        n_neg_keep = int(n_pos * UNDERSAMPLE_RATIO)

        if n_neg_keep < len(neg_ids):
            # Randomly select negatives
            neg_ids_keep = np.random.choice(neg_ids, size=n_neg_keep, replace=False)
            keep_indices = np.concatenate([pos_ids, neg_ids_keep])
            np.random.shuffle(keep_indices)

            X = X.iloc[keep_indices].reset_index(drop=True)
            y = y[keep_indices]
            ids = ids[keep_indices]
            print(f"Undersampled dataset shape: {X.shape}")
        else:
            print("Negative samples fewer than ratio, skipping undersampling.")

    else:
        # --- Transform using saved scaler ---
        if os.path.exists(SCALER_SAVE_PATH):
            print("Loading scaler and transforming data...")
            scaler = load(SCALER_SAVE_PATH)
            X[feature_names] = scaler.transform(X[feature_names])
        else:
            print(
                "Warning: Scaler not found. Skipping scaling (ensure train is run first)."
            )

    # 4. Save to Cache
    print(f"Caching processed data to {WORKING_DIR}...")
    X.to_parquet(cache_X)
    np.save(cache_y, y)
    np.save(cache_ids, ids)

    return X, y, ids
