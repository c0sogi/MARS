import os
import json
import hashlib
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import Timer, set_seed
from library.features_tracking import TrackingFeatureGenerator
from library.features_helmets import HelmetFeatureGenerator


class DataManager:
    """
    Orchestrates data loading, feature generation, caching, and formatting for the
    NFL Contact Detection pipeline. Implements the Multi-Modal Late-Fusion strategy.
    """

    def __init__(self):
        self.config = Config
        self.tracking_gen = TrackingFeatureGenerator()
        self.helmet_gen = HelmetFeatureGenerator()
        self.cache_dir = self.config.WORKING_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_config_hash(self) -> str:
        """
        Generates a hash based on relevant configuration parameters to ensure
        cache invalidation when settings change.
        """
        config_dict = {
            "WINDOW_MICRO": self.config.WINDOW_MICRO,
            "WINDOW_MACRO": self.config.WINDOW_MACRO,
            "TRACKING_BASE": self.config.TRACKING_BASE_COLS,
            "TRACKING_DERIVED": self.config.TRACKING_DERIVED_COLS,
            "TRACKING_INTERACTION": self.config.TRACKING_INTERACTION_COLS,
            "HELMET_BASE": self.config.HELMET_BASE_COLS,
            "HELMET_DERIVED": self.config.HELMET_DERIVED_COLS,
            "HELMET_INTERACTION": self.config.HELMET_INTERACTION_COLS,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def load_stream_data(
        self,
        stream: str,
        split: str,
        load_cached_data: bool = True,
        debug_sample: float = 1.0,
    ):
        """
        Loads prepared feature matrices (X), labels (y), and metadata (ids) for a specific stream and split.
        Handles caching based on configuration hash.

        Args:
            stream (str): 'tracking' (Stream A) or 'helmets' (Stream B).
            split (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to attempt loading from disk cache.
            debug_sample (float): Fraction of data to return (for debugging/testing). Default 1.0.

        Returns:
            tuple: (X, y, meta)
                - X (pd.DataFrame): Feature matrix.
                - y (np.array): Target labels (None for test if not available).
                - meta (pd.DataFrame): Metadata (contact_id, game_play, player_ids).
        """
        config_hash = self.get_config_hash()
        cache_prefix = os.path.join(self.cache_dir, f"{split}_{stream}_{config_hash}")

        path_X = f"{cache_prefix}_X.parquet"
        path_y = f"{cache_prefix}_y.npy"
        path_meta = f"{cache_prefix}_meta.parquet"

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(path_X) and os.path.exists(path_meta):
            # Check y existence (test split might not have it, or it might be cached as None/dummy)
            # For simplicity, we assume if X and meta exist, cache is valid.
            print(
                f"[{stream.upper()}] Loading {split} data from cache: {cache_prefix}..."
            )
            X = pd.read_parquet(path_X)
            meta = pd.read_parquet(path_meta)

            if os.path.exists(path_y):
                y = np.load(path_y)
            else:
                y = None

            # Apply debug sampling if requested
            if debug_sample < 1.0:
                return self._subsample(X, y, meta, debug_sample)
            return X, y, meta

        # 2. Generate from Scratch
        print(
            f"[{stream.upper()}] Cache miss or force reload. Generating {split} data..."
        )

        if stream == "tracking":
            df_raw = self.tracking_gen.generate_features(
                split, load_cached_data=load_cached_data
            )
        elif stream == "helmets":
            df_raw = self.helmet_gen.generate_features(
                split, load_cached_data=load_cached_data
            )
        else:
            raise ValueError(f"Unknown stream: {stream}")

        # 3. Format and Select Columns
        X, y, meta = self._format_data(df_raw, stream)

        # 4. Save to Cache
        print(f"[{stream.upper()}] Saving processed {split} data to cache...")
        X.to_parquet(path_X, index=False)
        meta.to_parquet(path_meta, index=False)
        if y is not None:
            np.save(path_y, y)

        # Apply debug sampling if requested
        if debug_sample < 1.0:
            return self._subsample(X, y, meta, debug_sample)

        return X, y, meta

    def _format_data(self, df: pd.DataFrame, stream: str):
        """
        Separates the raw dataframe into X (features), y (target), and meta (IDs).
        Enforces deterministic column ordering.
        """
        # Identify Metadata Columns
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        # Ensure we don't fail if some meta cols are missing (though they should be there)
        existing_meta = [c for c in meta_cols if c in df.columns]
        meta = df[existing_meta].copy()

        # Identify Target
        y = None
        if "contact" in df.columns:
            y = df["contact"].values

        # Identify Features
        expected_features = self._get_expected_columns(stream)

        # Check for missing columns and fill with 0 (robustness)
        missing_cols = [c for c in expected_features if c not in df.columns]
        if missing_cols:
            # Create a DataFrame of zeros for missing columns
            print(
                f"Warning: {len(missing_cols)} expected features missing. Filling with 0."
            )
            for c in missing_cols:
                df[c] = 0.0

        # Select and Reorder X
        X = df[expected_features].copy()

        # Downcast floats to save memory
        fcols = X.select_dtypes("float").columns
        X[fcols] = X[fcols].astype(np.float32)

        return X, y, meta

    def _get_expected_columns(self, stream: str) -> list:
        """
        Constructs the strict list of feature columns expected for the stream
        based on Config and FeatureGenerator logic.
        """
        cols = []
        micro_range = range(-self.config.WINDOW_MICRO, self.config.WINDOW_MICRO + 1)

        if stream == "tracking":
            # Logic mirrors TrackingFeatureGenerator
            # 1. Lags
            features_to_lag = [
                "x_position",
                "y_position",
                "speed",
                "acceleration",
                "sa",
                "sin_direction",
                "cos_direction",
                "sin_orientation",
                "cos_orientation",
                "v_x",
                "v_y",
            ]

            for lag in micro_range:
                suffix = f"_lag{lag}" if lag != 0 else ""
                # P1 features
                for f in features_to_lag:
                    cols.append(f"{f}{suffix}_p1")
                # P2 features
                for f in features_to_lag:
                    cols.append(f"{f}{suffix}_p2")
                # Interactions (computed at every lag)
                cols.append(f"distance{suffix}")
                cols.append(f"rel_speed{suffix}")

            # 2. Rolling (Macro)
            features_to_roll = ["speed", "acceleration"]
            for f in features_to_roll:
                cols.append(f"{f}_roll_mean_p1")
                cols.append(f"{f}_roll_std_p1")
                cols.append(f"{f}_roll_mean_p2")
                cols.append(f"{f}_roll_std_p2")

        elif stream == "helmets":
            # Logic mirrors HelmetFeatureGenerator
            features_to_lag = [
                "iou",
                "dist_centroids",
                "area_ratio",
                "area_p1",
                "area_p2",
            ]

            # Lag 0 (Original)
            for f in features_to_lag:
                cols.append(f)

            # Other Lags
            for lag in micro_range:
                if lag == 0:
                    continue
                suffix = f"_lag{lag}"
                for f in features_to_lag:
                    cols.append(f"{f}{suffix}")

        # Deduplicate while preserving order
        seen = set()
        unique_cols = []
        for c in cols:
            if c not in seen:
                unique_cols.append(c)
                seen.add(c)

        return unique_cols

    def _subsample(self, X, y, meta, frac):
        """
        Helper to subsample data for debugging/testing.
        """
        n = len(X)
        size = int(n * frac)
        indices = np.random.choice(n, size, replace=False)
        indices.sort()  # Keep temporal order roughly

        X_sub = X.iloc[indices].reset_index(drop=True)
        meta_sub = meta.iloc[indices].reset_index(drop=True)
        y_sub = y[indices] if y is not None else None

        return X_sub, y_sub, meta_sub

    def prepare_submission(self, test_meta, preds_prob, threshold=0.5):
        """
        Formats predictions into the submission DataFrame.

        Args:
            test_meta (pd.DataFrame): Metadata for test set (must contain contact_id).
            preds_prob (np.array): Predicted probabilities.
            threshold (float): Decision threshold.

        Returns:
            pd.DataFrame: Submission dataframe.
        """
        sub = pd.DataFrame()
        sub["contact_id"] = test_meta["contact_id"]
        sub["contact"] = (preds_prob >= threshold).astype(int)
        return sub
