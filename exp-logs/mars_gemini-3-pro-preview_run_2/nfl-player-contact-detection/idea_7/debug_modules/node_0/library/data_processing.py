import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class NFLContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Serves 3D tensors of shape (Batch, Window_Size, Features).
    """

    def __init__(self, features, labels=None, contact_ids=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.contact_ids = contact_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x


class FeatureEngineer:
    """
    Handles data loading, merging, ground imputation, and kinematic feature generation.
    """

    def __init__(self, config: Config):
        self.config = config
        self.window_size = config.window_size
        self.half_window = self.window_size // 2

    def load_tracking(self, split_name, relevant_games):
        """
        Loads and filters tracking data.
        """
        # Determine file name based on split (train/val use train_tracking, test uses test_tracking)
        filename = (
            "test_player_tracking.csv"
            if split_name == "test"
            else "train_player_tracking.csv"
        )
        path = os.path.join(self.config.input_dir, filename)

        # Load specific columns to save memory
        cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
        ]

        df = pd.read_csv(path, usecols=cols)

        # Filter to relevant games
        df = df[df["game_play"].isin(relevant_games)].copy()

        # Optimize dtypes
        df["step"] = df["step"].astype(np.int32)
        float_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
        ]
        for c in float_cols:
            df[c] = df[c].astype(np.float32)

        return df

    def expand_temporal_window(self, metadata_df):
        """
        Expands each row in metadata into 'window_size' rows (t-k ... t+k).
        """
        # Create offsets
        offsets = np.arange(-self.half_window, self.half_window + 1)

        # Repeat metadata rows
        # We use index to keep track of groups
        metadata_df = metadata_df.reset_index(drop=True)
        metadata_df["sample_id"] = metadata_df.index

        # Replicate
        expanded = metadata_df.loc[metadata_df.index.repeat(self.window_size)].copy()

        # Add offsets to step
        expanded["window_idx"] = np.tile(offsets, len(metadata_df))
        expanded["step"] = expanded["step"] + expanded["window_idx"]

        return expanded

    def process(self, metadata_df, tracking_df):
        """
        Main processing pipeline: Merge -> Impute -> Calculate Features -> Reshape.
        """
        # 1. Expand Metadata to Window
        print("  Expanding temporal windows...")
        expanded = self.expand_temporal_window(metadata_df)

        # 2. Prepare Player IDs for Merge
        # Ensure IDs are numeric where possible, keep 'G' as is for now
        # Tracking data IDs are numeric.

        # 3. Merge Player 1
        print("  Merging Player 1 tracking...")
        tracking_df_p1 = tracking_df.add_suffix("_1")
        expanded = expanded.merge(
            tracking_df_p1,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1"],
            how="left",
        )

        # 4. Merge Player 2
        print("  Merging Player 2 tracking...")
        # Handle 'G' temporarily by creating a numeric join column
        expanded["nfl_player_id_2_join"] = pd.to_numeric(
            expanded["nfl_player_id_2"], errors="coerce"
        )

        tracking_df_p2 = tracking_df.add_suffix("_2")
        expanded = expanded.merge(
            tracking_df_p2,
            left_on=["game_play", "step", "nfl_player_id_2_join"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
        )

        # Clean up merge cols
        drop_cols = [
            "game_play_1",
            "step_1",
            "game_play_2",
            "step_2",
            "nfl_player_id_2_join",
        ]
        expanded.drop(
            columns=[c for c in drop_cols if c in expanded.columns], inplace=True
        )

        # 5. Hybrid-Physics Ground Imputation
        print("  Imputing Ground physics...")
        is_ground = expanded["nfl_player_id_2"] == "G"

        # Position: Ground takes P1's position (Distance -> 0)
        expanded.loc[is_ground, "x_position_2"] = expanded.loc[
            is_ground, "x_position_1"
        ]
        expanded.loc[is_ground, "y_position_2"] = expanded.loc[
            is_ground, "y_position_1"
        ]

        # Kinematics: Ground has 0 velocity/accel
        expanded.loc[is_ground, "speed_2"] = 0.0
        expanded.loc[is_ground, "acceleration_2"] = 0.0
        expanded.loc[is_ground, "orientation_2"] = 0.0
        expanded.loc[is_ground, "direction_2"] = 0.0

        # Fill remaining NaNs (missing tracking data) with 0 or appropriate defaults
        # This prevents NaN propagation in features
        fill_cols = [
            c
            for c in expanded.columns
            if "position" in c
            or "speed" in c
            or "acceleration" in c
            or "orientation" in c
            or "direction" in c
        ]
        expanded[fill_cols] = expanded[fill_cols].fillna(0.0)

        # 6. Feature Engineering
        print("  Calculating explicit kinematic features...")

        # Is Ground Flag
        expanded["is_ground"] = is_ground.astype(np.float32)

        # Coordinates
        dx = expanded["x_position_1"] - expanded["x_position_2"]
        dy = expanded["y_position_1"] - expanded["y_position_2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Log Distance
        expanded["log_distance"] = np.log1p(dist)

        # Vector Velocity Components
        # Direction is in degrees, 0 is Y-axis (usually), need to check convention.
        # Standard NFL tracking: 0 is Y axis (short axis), 90 is X axis.
        # Convert to radians
        dir_rad_1 = np.radians(expanded["direction_1"])
        dir_rad_2 = np.radians(expanded["direction_2"])

        vx_1 = expanded["speed_1"] * np.sin(dir_rad_1)
        vy_1 = expanded["speed_1"] * np.cos(dir_rad_1)
        vx_2 = expanded["speed_2"] * np.sin(dir_rad_2)
        vy_2 = expanded["speed_2"] * np.cos(dir_rad_2)

        # Relative Speed (Vector magnitude difference)
        dvx = vx_1 - vx_2
        dvy = vy_1 - vy_2
        expanded["relative_speed"] = np.sqrt(dvx**2 + dvy**2)

        # Relative Acceleration (Scalar difference approximation as full vector accel is not provided)
        expanded["relative_acceleration"] = np.abs(
            expanded["acceleration_1"] - expanded["acceleration_2"]
        )

        # Closing Speed: - (v_rel . r_rel) / |r_rel|
        # r_rel = (dx, dy) pointing from 2 to 1?
        # dx = x1 - x2.
        # If moving towards each other, v1 is neg, v2 is pos?
        # Closing speed = rate of decrease of distance.
        # Projection: (vx_rel * dx + vy_rel * dy) / dist
        # If this dot product is negative, they are closing in.
        # We want positive value for closing speed.
        dot_prod = dvx * dx + dvy * dy
        # Clamp distance to avoid div/0
        clamped_dist = np.maximum(dist, 1e-6)
        # Closing speed: positive if closing
        expanded["closing_speed"] = -(dot_prod / clamped_dist)

        # Trig features for orientation/direction
        expanded["orientation_cos_1"] = np.cos(np.radians(expanded["orientation_1"]))
        expanded["orientation_sin_1"] = np.sin(np.radians(expanded["orientation_1"]))
        expanded["direction_cos_1"] = np.cos(dir_rad_1)
        expanded["direction_sin_1"] = np.sin(dir_rad_1)

        # 7. Reshape to (Batch, Window, Features)
        print("  Reshaping to 3D tensor...")
        feature_cols = self.config.feature_cols

        # Sort to ensure correct window order: sample_id, then window_idx (-5 to +5)
        expanded = expanded.sort_values(by=["sample_id", "window_idx"])

        # Extract numpy array
        # Shape: (N_samples * Window, N_features)
        flat_features = expanded[feature_cols].values.astype(np.float32)

        # Reshape
        num_samples = len(metadata_df)
        num_features = len(feature_cols)

        # Safety check
        if len(flat_features) != num_samples * self.window_size:
            print(
                f"Warning: Feature length mismatch. Expected {num_samples * self.window_size}, got {len(flat_features)}"
            )
            # Truncate or pad if necessary (shouldn't happen with logic above)

        X = flat_features.reshape(num_samples, self.window_size, num_features)

        return X


def load_and_process_data(split="train", debug=False, load_cached_data=True):
    """
    Orchestrates data loading and processing with caching.
    Args:
        split: 'train', 'validation', or 'test'
        debug: If True, subsets data.
        load_cached_data: If True, attempts to load from disk.
    Returns:
        dataset: NFLContactDataset
        metadata_df: The metadata dataframe (useful for IDs in test)
    """
    config = Config()

    # Define cache paths
    cache_prefix = f"{split}"
    if debug:
        cache_prefix += "_debug"

    path_X = os.path.join(config.artifact_dir, f"{cache_prefix}_X.npy")
    path_y = os.path.join(config.artifact_dir, f"{cache_prefix}_y.npy")
    path_ids = os.path.join(
        config.artifact_dir, f"{cache_prefix}_ids.npy"
    )  # For contact_ids

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(path_X):
        print(f"Loading cached {split} data from {config.artifact_dir}...")
        X = np.load(path_X)
        if split != "test":
            y = np.load(path_y)
        else:
            y = None

        # Load metadata to return alongside
        meta_path = os.path.join(config.metadata_dir, f"{split}.csv")
        metadata_df = pd.read_csv(meta_path)
        if debug:
            metadata_df = metadata_df.iloc[: config.debug_sample_size].copy()

        return NFLContactDataset(X, y, metadata_df["contact_id"].values), metadata_df

    # 2. Process from Scratch
    print(f"Processing {split} data from scratch...")

    # Load Metadata
    meta_path = os.path.join(config.metadata_dir, f"{split}.csv")
    metadata_df = pd.read_csv(meta_path)

    if debug:
        print(f"  Debug mode: Sampling {config.debug_sample_size} rows.")
        metadata_df = metadata_df.iloc[: config.debug_sample_size].copy()

    # Load Tracking
    relevant_games = metadata_df["game_play"].unique()
    fe = FeatureEngineer(config)
    tracking_df = fe.load_tracking(split, relevant_games)

    # Process Features
    X = fe.process(metadata_df, tracking_df)

    # Process Labels
    y = None
    if "contact" in metadata_df.columns and split != "test":
        y = metadata_df["contact"].values.astype(np.float32)

    # Save to Cache
    print(f"Saving {split} data to cache...")
    np.save(path_X, X)
    if y is not None:
        np.save(path_y, y)
    np.save(path_ids, metadata_df["contact_id"].values)

    # Clean up
    del tracking_df
    gc.collect()

    return NFLContactDataset(X, y, metadata_df["contact_id"].values), metadata_df
