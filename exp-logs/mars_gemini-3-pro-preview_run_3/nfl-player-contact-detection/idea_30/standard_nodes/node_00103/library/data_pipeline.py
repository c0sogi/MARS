import os
import pandas as pd
import numpy as np
import gc
from library.config import ProjectConfig
from library.utils import get_logger, generate_config_hash
from library.physics_engine import PhysicsEngine

logger = get_logger("DataPipeline")


class DataPipeline:
    """
    Orchestrates data loading, merging, preprocessing, and feature engineering
    for the Physically-Consistent Hybrid-Context Dual-Stream GBDT.
    """

    def __init__(self):
        self.config = ProjectConfig
        self.cache_dir = self.config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_paths(self, mode: str):
        """
        Returns the appropriate file paths based on the mode.
        """
        if mode == "train":
            meta_path = self.config.TRAIN_META_PATH
            tracking_path = self.config.TRAIN_TRACKING_PATH
            helmets_path = self.config.TRAIN_HELMETS_PATH
        elif mode == "validation":
            meta_path = self.config.VAL_META_PATH
            # Validation uses training raw data
            tracking_path = self.config.TRAIN_TRACKING_PATH
            helmets_path = self.config.TRAIN_HELMETS_PATH
        elif mode == "test":
            meta_path = self.config.TEST_META_PATH
            tracking_path = self.config.TEST_TRACKING_PATH
            helmets_path = self.config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return meta_path, tracking_path, helmets_path

    def _load_raw_data(self, mode: str):
        """
        Loads metadata and raw sensor data.
        """
        meta_path, tracking_path, helmets_path = self._get_paths(mode)

        logger.info(f"Loading metadata from {meta_path}")
        df_meta = pd.read_csv(meta_path)

        # Parse datetime if needed, though usually handled as strings or timestamps
        # df_meta['datetime'] = pd.to_datetime(df_meta['datetime'])

        logger.info(f"Loading tracking data from {tracking_path}")
        df_tracking = pd.read_csv(tracking_path)

        # Filter tracking data to relevant plays to save memory
        relevant_plays = df_meta["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

        logger.info(f"Loading helmets data from {helmets_path}")
        df_helmets = pd.read_csv(helmets_path)
        df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        return df_meta, df_tracking, df_helmets

    def _process_stream_a(
        self,
        df_labels: pd.DataFrame,
        df_tracking: pd.DataFrame,
        df_helmets: pd.DataFrame,
    ):
        """
        Process Stream A: Player-Player Interactions.
        """
        logger.info("Processing Stream A (Player-Player)...")

        # 1. Merge Tracking P1
        df_merged = pd.merge(
            df_labels,
            df_tracking.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # 2. Merge Tracking P2
        # Ensure nfl_player_id_2 is same type as tracking id
        # In metadata, nfl_player_id_2 is string (can be 'G'), tracking is numeric usually.
        # For Stream A, nfl_player_id_2 is a player ID.
        df_tracking_p2 = df_tracking.add_suffix("_p2")
        # We need to ensure join keys match types.
        # df_labels['nfl_player_id_2'] might be object.

        # Temporary conversion for merge
        df_merged["nfl_player_id_2_numeric"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        df_merged = pd.merge(
            df_merged,
            df_tracking_p2,
            left_on=["game_play", "step", "nfl_player_id_2_numeric"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # 3. Physics: Relational Metrics
        df_merged = PhysicsEngine.calculate_relational_metrics(df_merged)

        # 4. Visual Metrics
        df_merged = PhysicsEngine.calculate_visual_metrics(df_merged, df_helmets)

        # 5. Cross-Modal Verification Feature
        # visual_looming_mismatch = (Normalized Closure Rate - Visual Looming Rate)
        # Normalized Closure Rate approx closure_rate / distance (avoid div by zero)
        safe_dist = df_merged["distance"].replace(0, 0.1)
        norm_closure = df_merged["closure_rate"] / safe_dist
        df_merged["visual_looming_mismatch"] = (
            norm_closure - df_merged["visual_looming"]
        )

        # 6. Create Rolling Features (Cite Lesson 31)
        df_merged = PhysicsEngine.create_rolling_features(
            df_merged, ["distance"], self.config.ROLLING_WINDOW
        )

        # 7. Create Lagged Features
        # We need to lag all features defined in STREAM_A_FEATURES
        feats_to_lag = self.config.STREAM_A_FEATURES
        lags = self.config.LAGS

        df_final = PhysicsEngine.create_lagged_features(df_merged, feats_to_lag, lags)

        # 8. Select Final Columns
        # Construct the list of expected lagged column names
        final_cols = []
        for feat in feats_to_lag:
            for lag in lags:
                final_cols.append(f"{feat}_lag_{lag}")

        # Add identifiers and target
        id_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        target_col = ["contact"]

        # Validate Schema
        PhysicsEngine.validate_schema(df_final, final_cols)

        # Return only necessary columns
        return df_final[id_cols + final_cols + target_col].copy()

    def _process_stream_b(self, df_labels: pd.DataFrame, df_tracking: pd.DataFrame):
        """
        Process Stream B: Player-Ground Impacts.
        """
        logger.info("Processing Stream B (Player-Ground)...")

        # 1. Merge Tracking P1 (The player hitting the ground)
        # Note: STREAM_B_FEATURES expects standard names like 'speed', 'x_position', etc.
        # So we merge without suffix or rename after.
        # Let's merge and keep tracking names as is.
        df_merged = pd.merge(
            df_labels,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 2. Physics: Ego Dynamics
        # This adds v_surge, v_sway, a_surge, etc.
        df_merged = PhysicsEngine.calculate_ego_dynamics(df_merged)

        # 3. Physics: Angular Velocity (Cite Lesson 27)
        df_merged = PhysicsEngine.calculate_angular_velocity(df_merged)

        # 4. Create Rolling Features (Cite Lesson 31)
        df_merged = PhysicsEngine.create_rolling_features(
            df_merged, ["speed"], self.config.ROLLING_WINDOW
        )

        # 5. Create Lagged Features
        feats_to_lag = self.config.STREAM_B_FEATURES
        lags = self.config.LAGS

        df_final = PhysicsEngine.create_lagged_features(df_merged, feats_to_lag, lags)

        # 6. Select Final Columns
        final_cols = []
        for feat in feats_to_lag:
            for lag in lags:
                final_cols.append(f"{feat}_lag_{lag}")

        id_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        target_col = ["contact"]

        # Validate Schema
        PhysicsEngine.validate_schema(df_final, final_cols)

        return df_final[id_cols + final_cols + target_col].copy()

    def _undersample(self, df: pd.DataFrame):
        """
        Applies Targeted Majority Undersampling.
        Retains 100% of positives, subsamples negatives to NEG_POS_RATIO.
        """
        pos = df[df["contact"] == 1]
        neg = df[df["contact"] == 0]

        n_pos = len(pos)
        n_neg = len(neg)

        if n_pos == 0:
            return df  # Should not happen in training usually, but safe fallback

        n_keep = int(n_pos * self.config.NEG_POS_RATIO)

        if n_neg > n_keep:
            neg_sampled = neg.sample(n=n_keep, random_state=self.config.SEED)
            df_sampled = (
                pd.concat([pos, neg_sampled])
                .sample(frac=1, random_state=self.config.SEED)
                .reset_index(drop=True)
            )
            return df_sampled

        return df

    def run(self, mode: str = "train", load_cached_data: bool = True):
        """
        Main execution method.
        Checks cache -> Loads or Processes -> Saves -> Returns (X, y, ids) for both streams.

        Returns:
            dict: {
                'stream_a': {'X': ..., 'y': ..., 'ids': ...},
                'stream_b': {'X': ..., 'y': ..., 'ids': ...}
            }
        """
        config_hash = generate_config_hash()
        cache_prefix = f"{mode}_{config_hash}"

        path_a = os.path.join(self.cache_dir, f"stream_a_{cache_prefix}.parquet")
        path_b = os.path.join(self.cache_dir, f"stream_b_{cache_prefix}.parquet")

        # Check Cache
        if load_cached_data and os.path.exists(path_a) and os.path.exists(path_b):
            logger.info(
                f"Loading cached data from {self.cache_dir} (Hash: {config_hash})"
            )
            df_a = pd.read_parquet(path_a)
            df_b = pd.read_parquet(path_b)
        else:
            logger.info(
                f"Cache miss or force reload. Processing {mode} data from scratch..."
            )

            # Load Raw
            df_meta, df_tracking, df_helmets = self._load_raw_data(mode)

            # Split Streams
            # Stream B: player_id_2 == 'G'
            # Stream A: player_id_2 != 'G'
            mask_b = df_meta["nfl_player_id_2"] == "G"
            df_meta_b = df_meta[mask_b].copy()
            df_meta_a = df_meta[~mask_b].copy()

            # Process Streams
            df_a = self._process_stream_a(df_meta_a, df_tracking, df_helmets)
            df_b = self._process_stream_b(df_meta_b, df_tracking)

            # Undersample (Only for Train)
            if mode == "train":
                logger.info("Applying Targeted Majority Undersampling...")
                df_a = self._undersample(df_a)
                df_b = self._undersample(df_b)

            # Fill missing values with sentinel before saving
            df_a = df_a.fillna(self.config.MISSING_SENTINEL)
            df_b = df_b.fillna(self.config.MISSING_SENTINEL)

            # Save to Cache
            logger.info(f"Saving processed data to {self.cache_dir}")
            df_a.to_parquet(path_a)
            df_b.to_parquet(path_b)

            # Clean up raw data
            del df_meta, df_tracking, df_helmets, df_meta_a, df_meta_b
            gc.collect()

        # Prepare Output
        def extract_arrays(df, stream_type):
            if df.empty:
                return {"X": pd.DataFrame(), "y": pd.Series(), "ids": pd.DataFrame()}

            if stream_type == "A":
                feats = self.config.STREAM_A_FEATURES
            else:
                feats = self.config.STREAM_B_FEATURES

            lags = self.config.LAGS
            feature_cols = [f"{f}_lag_{l}" for f in feats for l in lags]

            X = df[feature_cols]
            y = df["contact"]
            ids = df[
                [
                    "contact_id",
                    "game_play",
                    "step",
                    "nfl_player_id_1",
                    "nfl_player_id_2",
                ]
            ]
            return {"X": X, "y": y, "ids": ids}

        result = {
            "stream_a": extract_arrays(df_a, "A"),
            "stream_b": extract_arrays(df_b, "B"),
        }

        logger.info(f"Data Pipeline ({mode}) Complete.")
        logger.info(f"Stream A Samples: {len(df_a)}")
        logger.info(f"Stream B Samples: {len(df_b)}")

        return result
