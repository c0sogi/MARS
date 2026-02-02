import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config, process_tracking_data


class NFLDataset(Dataset):
    """
    PyTorch Dataset for the NFL Contact Detection task.
    Handles loading of feature matrices and optional target labels.
    """

    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def impute_ground_and_engineer_features(merged_df):
    """
    Enforces strict physical rules for Ground contacts and computes relative kinematics.

    Logic:
    1. Ground Position Imputation: For Ground (G) contacts, impute the second entity's
       coordinates as the first entity's coordinates. This forces the geometric distance
       to 0, preserving the invariant "Contact implies Proximity".
    2. Ground Kinematics Imputation: Force Ground velocity and acceleration to 0.
       This preserves the "Relative Motion" signal (e.g., impact speed).
    3. Relative Features: Calculate Logarithmic Distance and Relative Speed for all lags.
    """
    # Identify ground contact rows
    is_ground = merged_df["nfl_player_id_2"] == "G"

    # Iterate through lag steps defined in Config
    for lag in Config.LAG_STEPS:
        x1 = f"x_position_lag{lag}_1"
        y1 = f"y_position_lag{lag}_1"
        x2 = f"x_position_lag{lag}_2"
        y2 = f"y_position_lag{lag}_2"
        s2 = f"speed_lag{lag}_2"
        a2 = f"acceleration_lag{lag}_2"

        # Impute Ground Physics
        # 1. Position: Impute Ground coordinates as Player's coordinates
        if x1 in merged_df.columns and x2 in merged_df.columns:
            merged_df.loc[is_ground, x2] = merged_df.loc[is_ground, x1]
        if y1 in merged_df.columns and y2 in merged_df.columns:
            merged_df.loc[is_ground, y2] = merged_df.loc[is_ground, y1]

        # 2. Kinematics: Impute Ground velocity and acceleration as 0
        if s2 in merged_df.columns:
            merged_df.loc[is_ground, s2] = 0.0
        if a2 in merged_df.columns:
            merged_df.loc[is_ground, a2] = 0.0

        # Compute Relative Features
        # Note: Non-ground NaNs (missing tracking) will result in NaNs here, handled later by fillna(0)
        if (
            x1 in merged_df.columns
            and x2 in merged_df.columns
            and y1 in merged_df.columns
            and y2 in merged_df.columns
        ):
            dx = merged_df[x1] - merged_df[x2]
            dy = merged_df[y1] - merged_df[y2]
            dist = np.sqrt(dx**2 + dy**2)
            merged_df[f"log_dist_lag{lag}"] = np.log1p(dist)

        if f"speed_lag{lag}_1" in merged_df.columns and s2 in merged_df.columns:
            merged_df[f"rel_speed_lag{lag}"] = (
                merged_df[f"speed_lag{lag}_1"] - merged_df[s2]
            )

    return merged_df


def build_dataset(mode="train", load_cached_data=True):
    """
    Orchestrates the data loading, processing, merging, and engineering pipeline.

    Args:
        mode (str): 'train' or 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame: The fully processed and merged dataset ready for scaling/training.
    """
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORK_DIR, f"{mode}_features.parquet")

    # 1. Cache Check
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. Load Metadata
    if mode == "train":
        meta_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
        val_meta_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))
        meta_df["is_val"] = 0
        val_meta_df["is_val"] = 1
        labels = pd.concat([meta_df, val_meta_df], ignore_index=True)

        if Config.DEBUG:
            labels = labels.sample(10000, random_state=Config.SEED).reset_index(
                drop=True
            )

        tracking_file = "train_player_tracking.csv"
    else:
        labels = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
        tracking_file = "test_player_tracking.csv"

    # 3. Load Tracking
    tracking_path = os.path.join(Config.INPUT_DIR, tracking_file)
    tracking = pd.read_csv(tracking_path)

    # Filter tracking to relevant plays to optimize memory usage
    relevant_gps = labels["game_play"].unique()
    tracking = tracking[tracking["game_play"].isin(relevant_gps)].copy()

    # 4. Process Tracking (Entity-Level Windowing)
    # Using imported function from library.config to generate temporal lags
    tracking_features = process_tracking_data(tracking)

    # 5. Merge Labels with Tracking (Entity-First)
    # Ensure numeric types for merging
    labels["nfl_player_id_1"] = pd.to_numeric(
        labels["nfl_player_id_1"], errors="coerce"
    )
    # nfl_player_id_2 contains 'G', so coerce to numeric for merging (G becomes NaN)
    labels["nfl_player_id_2_num"] = pd.to_numeric(
        labels["nfl_player_id_2"], errors="coerce"
    )

    # Merge Player 1
    merged = labels.merge(
        tracking_features,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P1 columns
    p1_cols = [
        c
        for c in tracking_features.columns
        if c not in ["game_play", "nfl_player_id", "step"]
    ]
    merged = merged.rename(columns={c: f"{c}_1" for c in p1_cols})

    # Merge Player 2
    merged = merged.merge(
        tracking_features,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    ).drop(columns=["nfl_player_id"])

    # Rename P2 columns
    merged = merged.rename(columns={c: f"{c}_2" for c in p1_cols})

    # 6. Hybrid Ground Imputation & Feature Engineering
    merged = impute_ground_and_engineer_features(merged)

    # 7. Cleanup
    drop_cols = [
        "path_endzone",
        "path_sideline",
        "path_all29",
        "datetime",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "nfl_player_id_2_num",
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # Fill remaining NaNs (e.g., missing tracking for players) with 0
    merged = merged.fillna(0)

    # 8. Save Cache
    merged.to_parquet(cache_path)

    return merged
