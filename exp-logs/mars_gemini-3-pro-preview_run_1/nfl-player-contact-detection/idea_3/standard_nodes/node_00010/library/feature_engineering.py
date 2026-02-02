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

    def create_unified_features(self, split="train", load_cached=True):
        """
        Generates Unified features for both Player-Player and Player-Ground contacts.
        Cite Lesson 00007: Unified Architectures with Type Indicators.
        """
        meta_path, track_path = self._get_paths(split)

        # Cache Config
        cache_config = {
            "split": split,
            "type": "Unified_v1",
            "features": self.config.STREAM_A_FEATURES,
            "window": self.config.WINDOW_SIZE,
        }
        path_x = get_hashed_path(f"{split}_features_unified", cache_config, ".parquet")

        if load_cached and os.path.exists(path_x):
            print(f"Loading cached Unified features for {split}...")
            return pd.read_parquet(path_x)

        print(f"Generating Unified features for {split}...")

        # 1. Load Metadata
        df_meta = pd.read_csv(meta_path)

        # Create Type Indicator (Cite Lesson 00007)
        df_meta["is_ground"] = (df_meta["nfl_player_id_2"] == "G").astype(int)

        # Ensure types for merge
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        # Handle 'G' in P2 by keeping as str
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
        df_merged.drop(columns=["nfl_player_id"], inplace=True, errors="ignore")

        track_cols = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_p1 = {c: f"{c}_p1" for c in track_cols}
        df_merged.rename(columns=rename_p1, inplace=True)

        # 4. Merge Player 2
        print("Merging Player 2 tracking...")
        # For 'G' rows, this merge will yield NaNs, which we then fill with 0
        # effectively "zeroing out" P2 features as per Lesson 00007.
        df_merged = pd.merge(
            df_merged,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_drop"),
        )
        df_merged.drop(columns=["nfl_player_id"], inplace=True, errors="ignore")

        rename_p2 = {c: f"{c}_p2" for c in track_cols}
        df_merged.rename(columns=rename_p2, inplace=True)

        # Fill NaNs for Ground (P2) with 0
        for col in rename_p2.values():
            df_merged[col] = df_merged[col].fillna(0)

        # 5. Compute Relative Features
        print("Computing relative interaction features...")

        lags = (
            [0]
            + [k for k in range(1, self.config.WINDOW_SIZE + 1)]
            + [f"minus_{k}" for k in range(1, self.config.WINDOW_SIZE + 1)]
        )

        feature_cols = ["is_ground"]

        for lag in lags:
            suffix = f"_{lag}"

            # Distance
            x1 = df_merged[f"x_position{suffix}_p1"]
            y1 = df_merged[f"y_position{suffix}_p1"]
            x2 = df_merged[f"x_position{suffix}_p2"]
            y2 = df_merged[f"y_position{suffix}_p2"]

            dist_col = f"distance{suffix}"
            # Standard dist
            dist_val = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # Handle Ground Distances: Set to NaN (Cite Lesson 00002)
            # This avoids model learning "distance to origin" as a feature
            is_ground_mask = df_merged["is_ground"] == 1
            dist_val[is_ground_mask] = np.nan

            df_merged[dist_col] = dist_val.fillna(999)  # Fill true missing with large
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

            # Orientation Diff
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
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        final_df = df_merged[meta_cols + feature_cols].copy()

        print(f"Saving Unified features to {path_x}...")
        save_cached_data(final_df, path_x)

        return final_df
