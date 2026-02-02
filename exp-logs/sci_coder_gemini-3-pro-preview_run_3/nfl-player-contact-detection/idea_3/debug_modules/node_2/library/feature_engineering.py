import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import get_logger

logger = get_logger("feature_engineering")


class FeatureEngineer:
    def __init__(self, load_cached_data=True):
        self.load_cached_data = load_cached_data
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_data(self, split="train"):
        """
        Loads data, generates features, and returns X, y, and ids.

        Args:
            split (str): One of 'train', 'validation', 'test'.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Target vector.
            ids (np.ndarray): Contact IDs.
        """
        # Define cache paths
        X_path = os.path.join(self.cache_dir, f"{split}_X.parquet")
        y_path = os.path.join(self.cache_dir, f"{split}_y.npy")
        ids_path = os.path.join(self.cache_dir, f"{split}_ids.npy")

        # Check cache
        if (
            self.load_cached_data
            and os.path.exists(X_path)
            and os.path.exists(y_path)
            and os.path.exists(ids_path)
        ):
            logger.info(f"Loading cached {split} data from {self.cache_dir}...")
            X = pd.read_parquet(X_path)
            y = np.load(y_path)
            ids = np.load(ids_path)
            return X, y, ids

        logger.info(f"Generating {split} data from scratch...")

        # 1. Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_META_PATH
            tracking_path = Config.TRAIN_TRACKING_PATH
        elif split == "validation":
            meta_path = Config.VAL_META_PATH
            tracking_path = Config.TRAIN_TRACKING_PATH
        else:  # test
            meta_path = Config.TEST_META_PATH
            tracking_path = Config.TEST_TRACKING_PATH

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Undersampling for training data
        if split == "train":
            pos = df_meta[df_meta["contact"] == 1]
            neg = df_meta[df_meta["contact"] == 0]

            n_pos = len(pos)
            n_neg_keep = int(n_pos * Config.UNDERSAMPLE_RATIO)

            if n_neg_keep < len(neg):
                neg = neg.sample(n=n_neg_keep, random_state=Config.SEED)
                logger.info(
                    f"Undersampled negatives: kept {n_neg_keep} out of {len(df_meta[df_meta['contact']==0])}"
                )

            df_meta = (
                pd.concat([pos, neg])
                .sample(frac=1, random_state=Config.SEED)
                .reset_index(drop=True)
            )
            logger.info(f"Train set shape after sampling: {df_meta.shape}")

        # 2. Load and Preprocess Tracking Data
        logger.info("Loading and preprocessing tracking data...")
        df_track = pd.read_csv(tracking_path)

        # Filter to relevant games
        relevant_games = df_meta["game_play"].unique()
        df_track = df_track[df_track["game_play"].isin(relevant_games)].copy()

        # Cyclical encoding
        df_track["sin_direction"] = np.sin(np.radians(df_track["direction"].fillna(0)))
        df_track["cos_direction"] = np.cos(np.radians(df_track["direction"].fillna(0)))
        df_track["sin_orientation"] = np.sin(
            np.radians(df_track["orientation"].fillna(0))
        )
        df_track["cos_orientation"] = np.cos(
            np.radians(df_track["orientation"].fillna(0))
        )

        # Fill missing values in tracking
        df_track = df_track.fillna(0)

        # 3. Generate Lagged Features
        logger.info("Generating lagged features...")
        base_features = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "sa",
            "sin_direction",
            "cos_direction",
            "sin_orientation",
            "cos_orientation",
        ]

        df_track.sort_values(["game_play", "nfl_player_id", "step"], inplace=True)

        # Use groupby shift to create lags
        lag_dfs = []
        window_range = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

        # Group by player to ensure shifts don't bleed across players
        g = df_track.groupby(["game_play", "nfl_player_id"])

        for k in window_range:
            # shift(-k) gets the value at t+k into row t
            shifted = g[base_features].shift(-k)
            shifted.columns = [f"{col}_lag_{k}" for col in base_features]
            lag_dfs.append(shifted)

        # Concatenate lags to the main tracking frame
        # We only need the keys and the lags
        df_track_wide = pd.concat(
            [df_track[["game_play", "nfl_player_id", "step"]]] + lag_dfs, axis=1
        )

        del df_track, lag_dfs, shifted
        gc.collect()

        # 4. Merge Metadata with Tracking
        logger.info("Merging metadata with tracking features...")

        # Ensure ID types match
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        df_track_wide["nfl_player_id"] = df_track_wide["nfl_player_id"].astype(str)

        # Prepare P1 columns
        p1_cols = {
            col: col.replace("_lag_", "_p1_lag_")
            for col in df_track_wide.columns
            if "_lag_" in col
        }

        # Merge P1
        df_merged = df_meta.merge(
            df_track_wide.rename(columns=p1_cols),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(
            columns=["nfl_player_id"]
        )  # Drop join key from right side

        # Prepare P2 columns
        p2_cols = {
            col: col.replace("_lag_", "_p2_lag_")
            for col in df_track_wide.columns
            if "_lag_" in col
        }

        # Merge P2
        df_merged = df_merged.merge(
            df_track_wide.rename(columns=p2_cols),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        del df_track_wide
        gc.collect()

        # 5. Handle Ground (P2='G') and Missing Data
        logger.info("Handling Ground contacts and missing data...")
        is_ground = df_merged["nfl_player_id_2"] == "G"

        for k in window_range:
            # Impute Ground position as P1 position (distance = 0)
            df_merged.loc[is_ground, f"x_position_p2_lag_{k}"] = df_merged.loc[
                is_ground, f"x_position_p1_lag_{k}"
            ]
            df_merged.loc[is_ground, f"y_position_p2_lag_{k}"] = df_merged.loc[
                is_ground, f"y_position_p1_lag_{k}"
            ]

            # Zero out kinematics for Ground
            kinematics = [
                "speed",
                "acceleration",
                "sa",
                "sin_direction",
                "cos_direction",
                "sin_orientation",
                "cos_orientation",
            ]
            for feat in kinematics:
                df_merged.loc[is_ground, f"{feat}_p2_lag_{k}"] = 0

        # Fill any remaining NaNs (missing tracking for players) with 0
        df_merged = df_merged.fillna(0)

        # 6. Compute Interaction Features
        logger.info("Computing interaction features...")
        for k in window_range:
            p1_x = df_merged[f"x_position_p1_lag_{k}"]
            p1_y = df_merged[f"y_position_p1_lag_{k}"]
            p2_x = df_merged[f"x_position_p2_lag_{k}"]
            p2_y = df_merged[f"y_position_p2_lag_{k}"]

            # Distance
            dx = p1_x - p2_x
            dy = p1_y - p2_y
            dist = np.sqrt(dx**2 + dy**2)
            df_merged[f"distance_lag_{k}"] = dist

            # Velocity components (approximate from speed & direction)
            v1_x = (
                df_merged[f"speed_p1_lag_{k}"] * df_merged[f"sin_direction_p1_lag_{k}"]
            )
            v1_y = (
                df_merged[f"speed_p1_lag_{k}"] * df_merged[f"cos_direction_p1_lag_{k}"]
            )
            v2_x = (
                df_merged[f"speed_p2_lag_{k}"] * df_merged[f"sin_direction_p2_lag_{k}"]
            )
            v2_y = (
                df_merged[f"speed_p2_lag_{k}"] * df_merged[f"cos_direction_p2_lag_{k}"]
            )

            dv_x = v1_x - v2_x
            dv_y = v1_y - v2_y

            # Relative Speed
            df_merged[f"rel_speed_lag_{k}"] = np.sqrt(dv_x**2 + dv_y**2)

            # Relative Acceleration
            df_merged[f"rel_acceleration_lag_{k}"] = (
                df_merged[f"acceleration_p1_lag_{k}"]
                - df_merged[f"acceleration_p2_lag_{k}"]
            )

            # Closure Rate: Project relative velocity onto distance vector
            # (v1 - v2) dot (p2 - p1) / |p2 - p1|
            # Vector p2 - p1 is (-dx, -dy)
            # dot = dv_x * (-dx) + dv_y * (-dy) = -(dv_x*dx + dv_y*dy)
            # closure = dot / dist
            dot_prod = -(dv_x * dx + dv_y * dy)

            # Avoid div by zero
            mask_valid = dist > 1e-6
            df_merged[f"closure_rate_lag_{k}"] = 0.0
            df_merged.loc[mask_valid, f"closure_rate_lag_{k}"] = (
                dot_prod[mask_valid] / dist[mask_valid]
            )

        # 7. Final Feature Selection and Flattening
        logger.info("Finalizing feature matrix...")
        final_cols = []

        for k in window_range:
            for feat in Config.PER_STEP_FEATURES:
                # Construct column name matching the dataframe
                # Interaction features have no suffix in Config, but have _lag_k in df
                if feat in [
                    "distance",
                    "rel_speed",
                    "rel_acceleration",
                    "closure_rate",
                ]:
                    col_name = f"{feat}_lag_{k}"
                else:
                    # Player features in Config have _p1 or _p2 suffix
                    # In df they are e.g. speed_p1_lag_k
                    # We simply append _lag_{k} to the Config name
                    col_name = f"{feat}_lag_{k}"

                final_cols.append(col_name)

        # Verify columns
        missing_cols = [c for c in final_cols if c not in df_merged.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataframe: {missing_cols[:5]}...")

        X = df_merged[final_cols].astype(np.float32)
        y = df_merged["contact"].values.astype(np.int8)
        ids = df_merged["contact_id"].values

        # Save to cache
        logger.info(f"Saving {split} data to cache...")
        X.to_parquet(X_path)
        np.save(y_path, y)
        np.save(ids_path, ids)

        logger.info(f"Completed {split} data processing. Shape: {X.shape}")
        return X, y, ids
