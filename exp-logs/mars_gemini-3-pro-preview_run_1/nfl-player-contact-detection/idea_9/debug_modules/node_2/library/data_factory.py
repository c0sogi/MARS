import pandas as pd
import numpy as np
import os
import gc
import library.config as config
from library.utils import CacheManager, setup_logging


class DataFactory:
    def __init__(self, mode="train"):
        """
        Args:
            mode (str): 'train', 'val', or 'test'. Determines which files to load.
        """
        self.mode = mode
        self.logger = setup_logging()
        self.cache = CacheManager(
            cache_dir=os.path.join(config.WORKING_DIR, "data_cache")
        )

        # Define paths based on mode
        if self.mode == "test":
            self.metadata_path = config.TEST_METADATA_PATH
            self.tracking_path = config.TEST_TRACKING_PATH
        elif self.mode == "val":
            self.metadata_path = config.VAL_METADATA_PATH
            self.tracking_path = config.TRAIN_TRACKING_PATH
        else:
            self.metadata_path = config.TRAIN_METADATA_PATH
            self.tracking_path = config.TRAIN_TRACKING_PATH

    def _load_raw_data(self):
        """Loads metadata and tracking data from CSVs."""
        self.logger.info(f"Loading metadata from {self.metadata_path}...")
        meta_df = pd.read_csv(self.metadata_path)

        self.logger.info(f"Loading tracking data from {self.tracking_path}...")
        track_df = pd.read_csv(self.tracking_path)

        # Optimize Tracking Data Types
        track_df["nfl_player_id"] = track_df["nfl_player_id"].astype("int32")
        track_df["step"] = track_df["step"].astype("int16")
        track_df["game_play"] = track_df["game_play"].astype("category")

        # Ensure metadata types match for merging
        meta_df["game_play"] = meta_df["game_play"].astype("category")
        meta_df["step"] = meta_df["step"].astype("int16")
        meta_df["nfl_player_id_1"] = meta_df["nfl_player_id_1"].astype("int32")

        return meta_df, track_df

    def _geometric_gating(self, meta_df, track_df):
        """
        Filters metadata to keep only plausible contacts.
        Rules:
        1. Keep ALL Player-Ground interactions.
        2. Keep Player-Player interactions where distance < GATING_DISTANCE.
        """
        self.logger.info("Applying Geometric Gating...")

        # Separate Ground and Player interactions
        is_ground = meta_df["nfl_player_id_2"] == "G"
        ground_df = meta_df[is_ground].copy()
        pp_df = meta_df[~is_ground].copy()

        if pp_df.empty:
            return ground_df

        # Convert P2 to int for merging
        pp_df["nfl_player_id_2"] = pp_df["nfl_player_id_2"].astype("int32")

        # Prepare minimal tracking for gating (current step only)
        track_cols = ["game_play", "step", "nfl_player_id", "x_position", "y_position"]
        t_mini = track_df[track_cols].copy()

        # Merge P1
        pp_df = (
            pp_df.merge(
                t_mini,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            .rename(columns={"x_position": "x1", "y_position": "y1"})
            .drop(columns=["nfl_player_id"])
        )

        # Merge P2
        pp_df = (
            pp_df.merge(
                t_mini,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            .rename(columns={"x_position": "x2", "y_position": "y2"})
            .drop(columns=["nfl_player_id"])
        )

        # Calculate Distance
        pp_df["dist"] = np.sqrt(
            (pp_df["x1"] - pp_df["x2"]) ** 2 + (pp_df["y1"] - pp_df["y2"]) ** 2
        )

        # Filter
        survivors = pp_df[pp_df["dist"] < config.GATING_DISTANCE].copy()

        # Drop temp columns
        survivors = survivors.drop(columns=["x1", "y1", "x2", "y2", "dist"])

        # Recombine with Ground
        # Ensure nfl_player_id_2 is object again to match ground_df
        survivors["nfl_player_id_2"] = survivors["nfl_player_id_2"].astype(str)

        final_df = pd.concat([ground_df, survivors], axis=0).reset_index(drop=True)
        self.logger.info(
            f"Gating Complete. Reduced {len(meta_df)} -> {len(final_df)} rows."
        )

        return final_df

    def _feature_engineering(self, meta_df, track_df):
        """
        Generates full-context features for the provided metadata.
        """
        self.logger.info(f"Starting Feature Engineering for {len(meta_df)} rows...")

        # 1. Preprocess Tracking Data
        # Sort for temporal calculations
        track_df = track_df.sort_values(["game_play", "nfl_player_id", "step"])

        # Calculate Derivatives
        # Group by player to respect boundaries
        grp = track_df.groupby(["game_play", "nfl_player_id"])

        # Jerk (Derivative of Acceleration)
        track_df["jerk"] = grp["acceleration"].diff().fillna(0).astype("float32")

        # Angular Jerk (Derivative of Orientation)
        # Handle circularity? Simple diff is usually sufficient for small steps,
        # but let's just take raw diff for speed.
        track_df["ang_jerk"] = grp["orientation"].diff().fillna(0).astype("float32")

        # Select features to merge
        feature_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "jerk",
            "ang_jerk",
        ]

        # 2. Prepare Metadata
        # Create a working copy
        df = meta_df.copy()

        # Handle P2 ID: Convert to numeric, set 'G' to NaN
        df["p2_int"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

        # 3. Windowed Merging
        # We merge tracking data for steps [t-W, ..., t+W]
        # W = FEATURE_WINDOW_SIZE (10)

        # To optimize, we'll iterate through the window offsets
        # For each offset, we merge P1 and P2 features

        for offset in range(
            -config.FEATURE_WINDOW_SIZE, config.FEATURE_WINDOW_SIZE + 1, 2
        ):
            # Step 2: Reduced resolution (every 2 steps) to save memory/time while keeping context
            # Or use full resolution if memory permits. 220GB is a lot. Let's do full resolution?
            # Let's do step=2 to be safe on column count (21 steps * 16 cols = 336 cols). Safe.
            pass

        # Let's use a stride of 2 for the window to reduce dimensionality
        # Window: -10, -8, ..., 0, ..., 8, 10
        offsets = list(
            range(-config.FEATURE_WINDOW_SIZE, config.FEATURE_WINDOW_SIZE + 1, 1)
        )

        # Pre-index tracking for faster merges
        # We can't easily pre-index in pandas merge, but ensuring dtypes helps.

        for offset in offsets:
            suffix = f"_t{offset}"

            # Calculate target step
            # We can't modify the tracking dataframe step, so we modify the merge key in metadata
            # But metadata 'step' varies.
            # Easiest: Create a temp key in metadata
            df["step_join"] = df["step"] + offset

            # Merge P1
            df = df.merge(
                track_df[["game_play", "step", "nfl_player_id"] + feature_cols],
                left_on=["game_play", "step_join", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_temp"),
            )
            # Rename and drop redundant
            rename_map = {c: f"p1_{c}{suffix}" for c in feature_cols}
            df = df.rename(columns=rename_map).drop(
                columns=["step_join", "nfl_player_id", "step_temp"], errors="ignore"
            )

            # Merge P2 (Only if not Ground)
            # We merge on p2_int. If p2_int is NaN (Ground), merge fails -> NaNs, which is correct.
            df["step_join"] = df["step"] + offset
            df = df.merge(
                track_df[["game_play", "step", "nfl_player_id"] + feature_cols],
                left_on=["game_play", "step_join", "p2_int"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_temp"),
            )
            rename_map = {c: f"p2_{c}{suffix}" for c in feature_cols}
            df = df.rename(columns=rename_map).drop(
                columns=["step_join", "nfl_player_id", "step_temp"], errors="ignore"
            )

            # 4. Compute Relative Features for this offset
            # If P2 is Ground (NaN features), we treat P2 as stationary at (0,0)?
            # No, Ground is relative to P1.
            # If P2 is Ground, we want "Distance" to be 0?
            # The prompt says "moments of contact".
            # Let's fill P2 NaNs with 0 for kinematics, but coordinates are tricky.
            # If P2 is Ground, we can't compute Euclidean distance from coordinates.
            # We will create a flag `is_ground` and let the tree handle the split.
            # We will fill NaNs with 0.

            p1_x = df[f"p1_x_position{suffix}"]
            p1_y = df[f"p1_y_position{suffix}"]
            p2_x = df[f"p2_x_position{suffix}"].fillna(0)  # Fill for calc
            p2_y = df[f"p2_y_position{suffix}"].fillna(0)

            # Distance
            # If Ground, set distance to -1 (sentinel) or 0?
            # Let's use a sentinel -1.0 so the model can split on it.
            # Actually, if we fill NaNs with 0, distance becomes sqrt(x1^2 + y1^2) which is distance to origin. Bad.
            # We need to mask distance for Ground.

            dist_col = f"dist{suffix}"
            df[dist_col] = np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

            # If P2 is Ground (p2_int is NaN), set distance to 0 (contact proxy) or -1.
            df.loc[df["p2_int"].isna(), dist_col] = -1.0

            # Relative Speed
            df[f"speed_diff{suffix}"] = df[f"p1_speed{suffix}"] - df[
                f"p2_speed{suffix}"
            ].fillna(0)

            # Clean up raw coordinates to enforce invariance?
            # The "Idea" says "Invariance: Strictly exclude absolute coordinates".
            # So we DROP x/y after computing distance.
            df = df.drop(
                columns=[
                    f"p1_x_position{suffix}",
                    f"p1_y_position{suffix}",
                    f"p2_x_position{suffix}",
                    f"p2_y_position{suffix}",
                ]
            )

        # Final Cleanup
        # Add is_ground feature
        df["is_ground"] = df["nfl_player_id_2"] == "G"
        df["is_ground"] = df["is_ground"].astype(int)

        # Drop helper
        df = df.drop(columns=["p2_int"])

        # Fill remaining NaNs (e.g. missing tracking data) with 0
        # This handles the P2='G' case for speed/accel columns implicitly
        df = df.fillna(0)

        self.logger.info(f"Feature Engineering Complete. Shape: {df.shape}")
        return df

    def get_train_dataset(self, load_cached=True):
        """
        Returns the Gated, Feature-Engineered Training Dataset.
        """

        def _compute():
            meta, track = self._load_raw_data()
            # 1. Gate
            gated_meta = self._geometric_gating(meta, track)
            # 2. Features
            features = self._feature_engineering(gated_meta, track)
            return features

        return self.cache.execute_with_cache(
            f"train_features_gated_{config.GATING_DISTANCE}y.parquet",
            _compute,
            load_cached_data=load_cached,
        )

    def get_val_dataset(self, load_cached=True):
        """
        Returns the FULL (Non-Gated) Validation Dataset with features.
        """

        def _compute():
            meta, track = self._load_raw_data()
            # No Gating for Validation to ensure accurate evaluation
            features = self._feature_engineering(meta, track)
            return features

        return self.cache.execute_with_cache(
            "val_features_full.parquet", _compute, load_cached_data=load_cached
        )

    def get_test_dataset(self, load_cached=True):
        """
        Returns the FULL (Non-Gated) Test Dataset with features.
        """

        def _compute():
            meta, track = self._load_raw_data()
            # No Gating for Test
            features = self._feature_engineering(meta, track)
            return features

        return self.cache.execute_with_cache(
            "test_features_full.parquet", _compute, load_cached_data=load_cached
        )
