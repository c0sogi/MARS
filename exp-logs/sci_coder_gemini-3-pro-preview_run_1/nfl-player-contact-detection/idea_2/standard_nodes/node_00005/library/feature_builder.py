import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.data_loader import load_metadata, load_tracking


class FeatureBuilder:
    """
    Constructs windowed tabular features for the Contact Detection task.
    Implements a sliding window approach to capture temporal dynamics
    for Gradient Boosting models.
    """

    def __init__(self):
        self.tracking_cols = Config.TRACKING_COLS
        self.derived_cols = Config.DERIVED_COLS
        self.window_pre = Config.WINDOW_PRE
        self.window_post = Config.WINDOW_POST

    def build_features(self, split="train", load_cached_data=True):
        """
        Main entry point to build features for a specific split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            pd.DataFrame: The processed feature dataframe.
        """
        # Determine cache path and tracking dataset type
        if split == "train":
            cache_path = Config.TRAIN_FEATURES_CACHE
            tracking_type = "train"
        elif split == "val":
            cache_path = Config.VAL_FEATURES_CACHE
            tracking_type = "train"
        elif split == "test":
            cache_path = Config.TEST_FEATURES_CACHE
            tracking_type = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} features from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # 2. Compute features from scratch
        print(f"Computing features for {split}...")

        # Load raw data
        df_meta = load_metadata(split=split)
        df_tracking = load_tracking(
            dataset_type=tracking_type, load_cached_data=load_cached_data
        )

        # Optimization: Filter tracking data to only relevant plays
        # This significantly reduces merge overhead
        relevant_plays = df_meta["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

        # Process features
        df_features = self._generate_windowed_features(df_meta, df_tracking)

        # 3. Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving {split} features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        # Clean up memory
        del df_meta, df_tracking
        gc.collect()

        return df_features

    def _generate_windowed_features(self, df_meta, df_tracking):
        """
        Iterates through the temporal window and constructs flattened features.
        """
        # Prepare base dataframe
        df_result = df_meta.copy()

        # Ensure join keys match types
        df_result["game_play"] = df_result["game_play"].astype(str)
        df_result["step"] = df_result["step"].astype(int)
        df_result["nfl_player_id_1"] = df_result["nfl_player_id_1"].astype(int)

        # Handle Player 2 ID: Convert to numeric for merging.
        # 'G' (Ground) becomes NaN in 'nfl_player_id_2_int'.
        df_result["nfl_player_id_2_int"] = pd.to_numeric(
            df_result["nfl_player_id_2"], errors="coerce"
        )

        # Identify Ground interactions for later imputation
        is_ground_mask = df_result["nfl_player_id_2"] == "G"

        # Add is_ground feature (useful for the model to distinguish context)
        df_result["is_ground"] = is_ground_mask.astype(int)

        # Prepare tracking dataframe for merging
        # We need (game_play, step, nfl_player_id) as keys
        track_cols_base = ["game_play", "step", "nfl_player_id"] + self.tracking_cols
        df_track_sub = df_tracking[track_cols_base].copy()

        # Iterate over the temporal window
        # Range: from -WINDOW_PRE to +WINDOW_POST (e.g., -10 to +10)
        lags = range(-self.window_pre, self.window_post + 1)

        for lag in lags:
            # Create a temporary step column for joining
            # We want tracking data at (current_step + lag)
            join_step_col = f"step_join_{lag}"
            df_result[join_step_col] = df_result["step"] + lag

            # --- Merge Player 1 ---
            p1_suffix = f"_p1_lag{lag}"

            # Prepare right-side dataframe with renamed columns to avoid collisions
            right_p1 = df_track_sub.copy()
            right_p1.columns = ["game_play", "step_r", "nfl_player_id_r"] + [
                f"{c}{p1_suffix}" for c in self.tracking_cols
            ]

            df_result = pd.merge(
                df_result,
                right_p1,
                left_on=["game_play", join_step_col, "nfl_player_id_1"],
                right_on=["game_play", "step_r", "nfl_player_id_r"],
                how="left",
            )
            # Drop the right-side join keys
            df_result.drop(columns=["step_r", "nfl_player_id_r"], inplace=True)

            # --- Merge Player 2 ---
            p2_suffix = f"_p2_lag{lag}"

            # Prepare right-side dataframe for P2
            right_p2 = df_track_sub.copy()
            right_p2.columns = ["game_play", "step_r", "nfl_player_id_r"] + [
                f"{c}{p2_suffix}" for c in self.tracking_cols
            ]

            df_result = pd.merge(
                df_result,
                right_p2,
                left_on=["game_play", join_step_col, "nfl_player_id_2_int"],
                right_on=["game_play", "step_r", "nfl_player_id_r"],
                how="left",
            )
            df_result.drop(columns=["step_r", "nfl_player_id_r"], inplace=True)

            # --- Handle Ground Logic for P2 ---
            # If P2 is Ground, set P2 tracking features to NaN (was 0.0)
            # This allows the model to distinguish 'Ground' from 'Zero Speed'
            # We use the 'is_ground' column which is safe inside the loop
            mask_ground = df_result["is_ground"] == 1
            p2_feature_cols = [f"{c}{p2_suffix}" for c in self.tracking_cols]
            df_result.loc[mask_ground, p2_feature_cols] = np.nan

            # --- Compute Derived Features ---
            d_suffix = f"_lag{lag}"

            # Distance: sqrt((x1-x2)^2 + (y1-y2)^2)
            x1 = df_result[f"x_position{p1_suffix}"]
            y1 = df_result[f"y_position{p1_suffix}"]
            x2 = df_result[f"x_position{p2_suffix}"]
            y2 = df_result[f"y_position{p2_suffix}"]

            # Calculate distance
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).astype("float32")

            # Set distance to NaN where P2 is ground to avoid arbitrary distance-to-origin
            dist[mask_ground] = np.nan

            df_result[f"distance{d_suffix}"] = dist

            # Speed Diff
            s1 = df_result[f"speed{p1_suffix}"]
            s2 = df_result[f"speed{p2_suffix}"]
            df_result[f"speed_diff{d_suffix}"] = np.abs(s1 - s2).astype("float32")

            # Accel Diff
            a1 = df_result[f"acceleration{p1_suffix}"]
            a2 = df_result[f"acceleration{p2_suffix}"]
            df_result[f"acc_diff{d_suffix}"] = np.abs(a1 - a2).astype("float32")

            # Direction Diff (Angular)
            dir1 = df_result[f"direction{p1_suffix}"]
            dir2 = df_result[f"direction{p2_suffix}"]
            diff_dir = np.abs(dir1 - dir2)
            df_result[f"dir_diff{d_suffix}"] = np.minimum(
                diff_dir, 360 - diff_dir
            ).astype("float32")

            # Orientation Diff (Angular)
            ori1 = df_result[f"orientation{p1_suffix}"]
            ori2 = df_result[f"orientation{p2_suffix}"]
            diff_ori = np.abs(ori1 - ori2)
            df_result[f"orient_diff{d_suffix}"] = np.minimum(
                diff_ori, 360 - diff_ori
            ).astype("float32")

            # Drop the temporary join step column
            df_result.drop(columns=[join_step_col], inplace=True)

        # Cleanup intermediate columns
        df_result.drop(columns=["nfl_player_id_2_int"], inplace=True, errors="ignore")

        return df_result
