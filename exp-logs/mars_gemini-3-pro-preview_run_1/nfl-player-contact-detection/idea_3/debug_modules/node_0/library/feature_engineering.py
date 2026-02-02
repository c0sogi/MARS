import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import (
    get_hashed_path,
    load_cached_data,
    save_cached_data,
    print_metric,
)


class FeatureEngineer:
    """
    Handles feature engineering for the Split-Stream Kinematic Architecture.
    Generates distinct datasets for the Interaction Model (Stream A) and Impact Model (Stream B).
    """

    def __init__(self, config=Config):
        self.config = config

    def _get_paths(self, split):
        """Resolves metadata and tracking paths based on the split."""
        if split == "train":
            return self.config.TRAIN_METADATA_PATH, self.config.TRAIN_TRACKING_PATH
        elif split == "val":
            return self.config.VAL_METADATA_PATH, self.config.TRAIN_TRACKING_PATH
        elif split == "test":
            return self.config.TEST_METADATA_PATH, self.config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

    def _process_tracking_data(self, tracking_path, load_cached=True):
        """
        Loads tracking data, computes derived features (Jerk), and creates a 'wide' format
        containing lag features for the defined window size.
        """
        # Config for caching key
        cache_config = {
            "window_size": self.config.WINDOW_SIZE,
            "tracking_cols": self.config.TRACKING_COLS,
            "source": os.path.basename(tracking_path),
        }

        # Define cache path
        cache_path = get_hashed_path("tracking_wide", cache_config, ".parquet")

        # 1. Try Load Cache
        if load_cached:
            data = load_cached_data(cache_path)
            if data is not None:
                print(f"Loaded cached wide tracking data from {cache_path}")
                return data

        print(f"Computing wide tracking data from {tracking_path}...")

        # 2. Load Raw Data
        df = pd.read_csv(tracking_path)

        # Ensure types
        df["nfl_player_id"] = df["nfl_player_id"].astype(str)
        df["game_play"] = df["game_play"].astype(str)
        df["step"] = df["step"].astype(int)

        # 3. Derived Features (Before Windowing)
        # Jerk = d(Acceleration) / dt. Data is 10Hz, so dt=0.1.
        # However, we just want the delta, scaling doesn't matter for trees/CNNs as much as the signal.
        # We group by game_play and nfl_player_id to ensure we don't diff across boundaries.
        print("Computing derived features (Jerk)...")
        # Sorting is critical for shift/diff
        df = df.sort_values(["game_play", "nfl_player_id", "step"])

        # Vectorized diff: We can just take diff of the column, then mask out the start of groups
        # This is faster than groupby().diff() for large dataframes
        df["jerk"] = df["acceleration"].diff().fillna(0)

        # Mask out boundaries where game_play or player changed
        # If row i and row i-1 have different game_play or player, jerk is invalid (0)
        mask = (df["game_play"] != df["game_play"].shift(1)) | (
            df["nfl_player_id"] != df["nfl_player_id"].shift(1)
        )
        df.loc[mask, "jerk"] = 0

        # 4. Create Wide Format (Lags)
        # We need features for [t - W, ..., t + W]
        # We will pivot this so one row = one step containing all temporal context

        features_to_lag = self.config.TRACKING_COLS + ["jerk"]
        window_size = self.config.WINDOW_SIZE

        # We will collect dataframes and concat (more memory efficient than repeated merges)
        lagged_dfs = []

        print(f"Generating lag features for window +/- {window_size}...")

        # Include the base step (lag 0)
        # We rename columns to f"{col}_0" for consistency
        base_df = df[["game_play", "nfl_player_id", "step"] + features_to_lag].copy()
        base_df.columns = ["game_play", "nfl_player_id", "step"] + [
            f"{c}_0" for c in features_to_lag
        ]
        lagged_dfs.append(base_df)

        # Generate lags
        # Positive lag k means looking at step t+k (future relative to t)
        # Negative lag k means looking at step t-k (past relative to t)
        # We use shift(-k) to bring future data to current row

        for k in range(1, window_size + 1):
            # Future (t + k): shift(-k)
            future_df = df[features_to_lag].shift(-k)
            future_df.columns = [f"{c}_{k}" for c in features_to_lag]

            # Past (t - k): shift(k)
            past_df = df[features_to_lag].shift(k)
            past_df.columns = [f"{c}_minus_{k}" for c in features_to_lag]

            # Handle boundary masking for shifts
            # For shift(-k), we must ensure row i and row i+k are same group
            # For shift(k), we must ensure row i and row i-k are same group
            # We can check game_play/player equality

            # Optimization: Instead of complex masking, we can just concat and then clean up
            # based on the fact that we will merge on (game_play, step, player) later.
            # If we shift data from another play into this row, it's fine as long as we don't use it?
            # NO, we must be correct.

            # Robust way:
            # Check group validity
            group_ids = df["game_play"] + "_" + df["nfl_player_id"]

            # Mask Future
            valid_future = group_ids == group_ids.shift(-k)
            future_df[~valid_future] = 0  # or NaN

            # Mask Past
            valid_past = group_ids == group_ids.shift(k)
            past_df[~valid_past] = 0  # or NaN

            lagged_dfs.append(future_df)
            lagged_dfs.append(past_df)

        # Concatenate horizontally
        # Since all DFs are aligned by index (original df index), this works
        print("Concatenating wide tracking data...")
        wide_df = pd.concat(lagged_dfs, axis=1)

        # Remove duplicate columns if any (game_play, etc are only in the first one)
        # Actually, we only kept keys in the first one.

        # Reduce memory
        wide_df = wide_df.astype(np.float32, errors="ignore")
        # Restore keys to object/int
        wide_df["game_play"] = df["game_play"]
        wide_df["nfl_player_id"] = df["nfl_player_id"]
        wide_df["step"] = df["step"]

        print(f"Wide tracking shape: {wide_df.shape}")

        # Cache
        save_cached_data(wide_df, cache_path)

        return wide_df

    def create_stream_a_features(self, split="train", load_cached=True):
        """
        Generates features for Stream A (Interaction Model).
        Target: Player-Player contacts.
        Features: Flattened vector of relative kinematics.
        """
        meta_path, track_path = self._get_paths(split)

        # Cache Config
        cache_config = {
            "split": split,
            "type": "StreamA",
            "features": self.config.STREAM_A_FEATURES,
            "window": self.config.WINDOW_SIZE,
        }
        path_x = get_hashed_path(f"{split}_features_streamA", cache_config, ".parquet")

        if load_cached and os.path.exists(path_x):
            print(f"Loading cached Stream A features for {split}...")
            return pd.read_parquet(path_x)

        print(f"Generating Stream A features for {split}...")

        # 1. Load Metadata
        df_meta = pd.read_csv(meta_path)

        # Filter for Player-Player only
        # nfl_player_id_2 != 'G'
        df_meta = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()

        # Ensure types for merge
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)

        # 2. Load Wide Tracking
        df_track = self._process_tracking_data(track_path, load_cached=load_cached)

        # 3. Merge Player 1
        print("Merging Player 1 tracking...")
        df_merged = pd.merge(
            df_meta,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_drop"),
        )
        # Drop redundant keys
        df_merged.drop(columns=["nfl_player_id"], inplace=True, errors="ignore")

        # Rename columns to _p1 suffix
        # Columns in df_track (except keys) need renaming
        track_cols = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_p1 = {c: f"{c}_p1" for c in track_cols}
        df_merged.rename(columns=rename_p1, inplace=True)

        # 4. Merge Player 2
        print("Merging Player 2 tracking...")
        df_merged = pd.merge(
            df_merged,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_drop"),
        )
        df_merged.drop(columns=["nfl_player_id"], inplace=True, errors="ignore")

        # Rename columns to _p2 suffix
        rename_p2 = {c: f"{c}_p2" for c in track_cols}
        df_merged.rename(columns=rename_p2, inplace=True)

        # 5. Compute Relative Features
        print("Computing relative interaction features...")

        # Identify all lag suffixes: _0, _1, _minus_1, etc.
        # We iterate through the window range to reconstruct the suffixes
        lags = (
            [0]
            + [k for k in range(1, self.config.WINDOW_SIZE + 1)]
            + [f"minus_{k}" for k in range(1, self.config.WINDOW_SIZE + 1)]
        )

        feature_cols = []

        for lag in lags:
            suffix = f"_{lag}"

            # Distance
            x1 = df_merged[f"x_position{suffix}_p1"]
            y1 = df_merged[f"y_position{suffix}_p1"]
            x2 = df_merged[f"x_position{suffix}_p2"]
            y2 = df_merged[f"y_position{suffix}_p2"]

            dist_col = f"distance{suffix}"
            df_merged[dist_col] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).fillna(
                999
            )  # Fill NaN with large dist
            feature_cols.append(dist_col)

            # Speed Diff
            s1 = df_merged[f"speed{suffix}_p1"]
            s2 = df_merged[f"speed{suffix}_p2"]
            sd_col = f"speed_diff{suffix}"
            df_merged[sd_col] = np.abs(s1 - s2).fillna(0)
            feature_cols.append(sd_col)

            # Acc Diff
            a1 = df_merged[f"acceleration{suffix}_p1"]
            a2 = df_merged[f"acceleration{suffix}_p2"]
            ad_col = f"acc_diff{suffix}"
            df_merged[ad_col] = np.abs(a1 - a2).fillna(0)
            feature_cols.append(ad_col)

            # Orientation Diff (Circular diff)
            o1 = df_merged[f"orientation{suffix}_p1"]
            o2 = df_merged[f"orientation{suffix}_p2"]
            od_col = f"orientation_diff{suffix}"
            diff = np.abs(o1 - o2)
            df_merged[od_col] = np.minimum(diff, 360 - diff).fillna(0)
            feature_cols.append(od_col)

            # Direction Diff
            d1 = df_merged[f"direction{suffix}_p1"]
            d2 = df_merged[f"direction{suffix}_p2"]
            dd_col = f"direction_diff{suffix}"
            diff_d = np.abs(d1 - d2)
            df_merged[dd_col] = np.minimum(diff_d, 360 - diff_d).fillna(0)
            feature_cols.append(dd_col)

        # Select final columns
        # Include identifiers and target
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        final_df = df_merged[meta_cols + feature_cols].copy()

        # Save
        print(f"Saving Stream A features to {path_x}...")
        save_cached_data(final_df, path_x)

        return final_df

    def create_stream_b_features(self, split="train", load_cached=True):
        """
        Generates features for Stream B (Impact Model).
        Target: Player-Ground contacts.
        Features: 3D Tensor (N, Channels, Time) of absolute kinematics.
        Returns:
            X: numpy array (N, C, L)
            metadata: DataFrame with IDs and targets
        """
        meta_path, track_path = self._get_paths(split)

        # Cache Config
        cache_config = {
            "split": split,
            "type": "StreamB",
            "features": self.config.STREAM_B_FEATURES,
            "window": self.config.WINDOW_SIZE,
        }
        path_x = get_hashed_path(f"{split}_features_streamB_X", cache_config, ".npy")
        path_meta = get_hashed_path(
            f"{split}_features_streamB_meta", cache_config, ".parquet"
        )

        if load_cached and os.path.exists(path_x) and os.path.exists(path_meta):
            print(f"Loading cached Stream B features for {split}...")
            X = load_cached_data(path_x)
            meta = load_cached_data(path_meta)
            return X, meta

        print(f"Generating Stream B features for {split}...")

        # 1. Load Metadata
        df_meta = pd.read_csv(meta_path)

        # Filter for Player-Ground only
        df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        # Ensure types
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)

        # 2. Load Wide Tracking
        df_track = self._process_tracking_data(track_path, load_cached=load_cached)

        # 3. Merge Player 1 (The only player)
        print("Merging Player 1 tracking...")
        df_merged = pd.merge(
            df_meta,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 4. Construct Tensor
        # We need to extract columns in temporal order: t-10, ..., t, ..., t+10
        # Channels: speed, acceleration, jerk, orientation, direction, sa

        channels = self.config.STREAM_B_FEATURES  # e.g. ['speed', 'acceleration', ...]
        window = self.config.WINDOW_SIZE

        # Order of lags: minus_10, minus_9, ..., 0, ..., 10
        # Note: In wide tracking generation, we named them:
        # 0 -> "{c}_0"
        # k -> "{c}_{k}"
        # -k -> "{c}_minus_{k}"

        # Construct ordered list of suffixes
        ordered_suffixes = []
        for k in range(window, 0, -1):
            ordered_suffixes.append(f"_minus_{k}")
        ordered_suffixes.append("_0")
        for k in range(1, window + 1):
            ordered_suffixes.append(f"_{k}")

        seq_len = len(ordered_suffixes)  # 21
        num_channels = len(channels)
        num_samples = len(df_merged)

        print(
            f"Constructing tensor with shape ({num_samples}, {num_channels}, {seq_len})..."
        )

        X = np.zeros((num_samples, num_channels, seq_len), dtype=np.float32)

        for c_idx, channel in enumerate(channels):
            for t_idx, suffix in enumerate(ordered_suffixes):
                col_name = f"{channel}{suffix}"
                # Fill NaN with 0
                X[:, c_idx, t_idx] = df_merged[col_name].fillna(0).values

        # 5. Prepare Metadata Output
        meta_out = df_merged[
            ["contact_id", "game_play", "step", "nfl_player_id_1", "contact"]
        ].copy()

        # Save
        print(f"Saving Stream B features to {path_x}...")
        save_cached_data(X, path_x)
        save_cached_data(meta_out, path_meta)

        return X, meta_out
