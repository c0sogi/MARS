import pandas as pd
import numpy as np
import os
import gc
from typing import Tuple, List, Dict
from library.config import (
    WORKING_DIR,
    LAG_OFFSETS,
    STREAM_A_FEATURES,
    STREAM_B_FEATURES,
    SEED,
)
from library.physics_engine import (
    calculate_euclidean_distance,
    calculate_closure_rate,
    project_ego_velocity,
    calculate_derivatives,
)
from library.utils import reduce_mem_usage, verify_schema


class FeatureBuilder:
    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.cache_dir = os.path.join(WORKING_DIR, "features")
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_features(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame, load_cached: bool = True
    ) -> Tuple[Dict, Dict]:
        """
        Orchestrates the creation of features for both streams.

        Returns:
            Two dictionaries (data_a, data_b), each containing:
            {
                'X': pd.DataFrame (features),
                'y': np.ndarray (targets),
                'ids': np.ndarray (contact_ids)
            }
        """
        print(f"[{self.mode.upper()}] Building features...")

        # Define cache filenames
        cache_files = {
            "A": {
                "X": os.path.join(self.cache_dir, f"stream_a_X_{self.mode}.parquet"),
                "y": os.path.join(self.cache_dir, f"stream_a_y_{self.mode}.npy"),
                "ids": os.path.join(self.cache_dir, f"stream_a_ids_{self.mode}.npy"),
            },
            "B": {
                "X": os.path.join(self.cache_dir, f"stream_b_X_{self.mode}.parquet"),
                "y": os.path.join(self.cache_dir, f"stream_b_y_{self.mode}.npy"),
                "ids": os.path.join(self.cache_dir, f"stream_b_ids_{self.mode}.npy"),
            },
        }

        # Check cache existence
        cache_exists_a = all(os.path.exists(p) for p in cache_files["A"].values())
        cache_exists_b = all(os.path.exists(p) for p in cache_files["B"].values())

        data_a = {}
        data_b = {}

        # --- Load/Build Stream A ---
        if load_cached and cache_exists_a:
            print("Loading Stream A features from cache...")
            data_a["X"] = pd.read_parquet(cache_files["A"]["X"])
            data_a["y"] = np.load(cache_files["A"]["y"])
            data_a["ids"] = np.load(cache_files["A"]["ids"], allow_pickle=True)
        else:
            print("Constructing Stream A features...")
            if not df_a.empty:
                data_a = self._build_stream_a(df_a)
                # Save to cache
                print("Saving Stream A to cache...")
                data_a["X"].to_parquet(cache_files["A"]["X"], index=False)
                np.save(cache_files["A"]["y"], data_a["y"])
                np.save(cache_files["A"]["ids"], data_a["ids"])
            else:
                # Handle empty case (e.g. rare edge case in subsetting)
                data_a = {"X": pd.DataFrame(), "y": np.array([]), "ids": np.array([])}

        # --- Load/Build Stream B ---
        if load_cached and cache_exists_b:
            print("Loading Stream B features from cache...")
            data_b["X"] = pd.read_parquet(cache_files["B"]["X"])
            data_b["y"] = np.load(cache_files["B"]["y"])
            data_b["ids"] = np.load(cache_files["B"]["ids"], allow_pickle=True)
        else:
            print("Constructing Stream B features...")
            if not df_b.empty:
                data_b = self._build_stream_b(df_b)
                # Save to cache
                print("Saving Stream B to cache...")
                data_b["X"].to_parquet(cache_files["B"]["X"], index=False)
                np.save(cache_files["B"]["y"], data_b["y"])
                np.save(cache_files["B"]["ids"], data_b["ids"])
            else:
                data_b = {"X": pd.DataFrame(), "y": np.array([]), "ids": np.array([])}

        return data_a, data_b

    def _apply_temporal_pyramids(
        self, df: pd.DataFrame, feature_cols: List[str], group_cols: List[str]
    ) -> pd.DataFrame:
        """
        Flattens features over time using exponential lags.
        """
        # Ensure data is sorted by time
        df = df.sort_values(by=group_cols + ["step"]).copy()

        # We will collect all feature dataframes here
        dfs_to_concat = []

        # Base features (lag 0)
        # We process lag 0 separately or as part of the loop.
        # The LAG_OFFSETS list includes 0, so we handle it in the loop.

        # Group object for shifting
        grouped = df.groupby(group_cols)

        for lag in LAG_OFFSETS:
            # Shift features
            if lag == 0:
                shifted = df[feature_cols].copy()
            else:
                shifted = grouped[feature_cols].shift(lag)

            # Rename columns
            suffix = f"_lag{lag}" if lag != 0 else ""
            shifted.columns = [f"{col}{suffix}" for col in feature_cols]

            dfs_to_concat.append(shifted)

        # Concatenate all lagged features
        df_flattened = pd.concat(dfs_to_concat, axis=1)

        return df_flattened

    def _build_stream_a(self, df: pd.DataFrame) -> Dict:
        """
        Builds Interaction Model features (Geometry + Energy + Vision).
        """
        # 1. Feature Engineering

        # Euclidean Distance
        df["distance"] = calculate_euclidean_distance(
            df["x_position_p1"],
            df["y_position_p1"],
            df["x_position_p2"],
            df["y_position_p2"],
        )

        # Closure Rate (requires temporal diff, group by pair)
        # Group by game_play, player1, player2
        group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2_int"]
        df = df.sort_values(by=group_cols + ["step"])

        df["closure_rate"] = df.groupby(group_cols)["distance"].transform(
            lambda x: calculate_closure_rate(x)
        )

        # Ensure System Energy columns exist (renaming if necessary or verifying)
        # DataManager provides: speed_p1, speed_p2, acceleration_p1, acceleration_p2

        # 2. Select Base Features
        # STREAM_A_FEATURES defined in config:
        # [distance, closure_rate, speed_p1, speed_p2, acc_p1, acc_p2, iou_sideline, ...]

        # Verify schema before processing
        # Note: 'iou_sideline' etc are created in DataManager
        req_cols = STREAM_A_FEATURES + group_cols + ["step", "contact_id", "contact"]
        # Filter to available columns (contact might be missing in test if not provided, but metadata has it as placeholder)

        # Impute Visual Consensus features with -999 as per requirements
        visual_cols = ["iou_sideline", "iou_endzone", "iou_max", "iou_min", "iou_diff"]
        for col in visual_cols:
            if col in df.columns:
                df[col] = df[col].fillna(-999)

        # 3. Apply Temporal Pyramids
        print(f"Applying Temporal Pyramids for Stream A ({len(df)} rows)...")
        X_flattened = self._apply_temporal_pyramids(df, STREAM_A_FEATURES, group_cols)

        # 4. Final Cleanup
        # Fill NaNs generated by shifting (lags at edges of play)
        # For visual features, we keep -999. For others, XGBoost handles NaN,
        # but to be safe/clean we can fill with 0 or leave as NaN.
        # The requirement said "Impute missing values with -999" for visual.
        # We'll leave physics NaNs for XGBoost (better than 0 which implies stop).

        # Re-apply -999 for lagged visual columns if they became NaN due to shift
        for col in X_flattened.columns:
            if any(v in col for v in visual_cols):
                X_flattened[col] = X_flattened[col].fillna(-999)

        # Extract Target and IDs
        y = df["contact"].values.astype(int)
        ids = df["contact_id"].values

        # Reduce memory
        X_flattened = reduce_mem_usage(X_flattened, verbose=False)

        return {"X": X_flattened, "y": y, "ids": ids}

    def _build_stream_b(self, df: pd.DataFrame) -> Dict:
        """
        Builds Impact Model features (Hybrid Context + Ego-Physics).
        """
        # 1. Feature Engineering (Physics)

        # Grouping for derivatives
        group_cols = ["game_play", "nfl_player_id_1"]
        df = df.sort_values(by=group_cols + ["step"])

        # Project Velocity -> Surge/Sway
        v_surge, v_sway = project_ego_velocity(
            df["speed"], df["direction"], df["orientation"]
        )
        df["v_surge"] = v_surge
        df["v_sway"] = v_sway

        # Calculate Acceleration (Derivatives of Velocity)
        # Using transform to keep shape aligned
        df["a_surge"] = df.groupby(group_cols)["v_surge"].transform(
            lambda x: calculate_derivatives(x)
        )
        df["a_sway"] = df.groupby(group_cols)["v_sway"].transform(
            lambda x: calculate_derivatives(x)
        )

        # Calculate Jerk (Derivatives of Acceleration)
        df["j_surge"] = df.groupby(group_cols)["a_surge"].transform(
            lambda x: calculate_derivatives(x)
        )
        df["j_sway"] = df.groupby(group_cols)["a_sway"].transform(
            lambda x: calculate_derivatives(x)
        )

        # 2. Select Base Features
        # STREAM_B_FEATURES: [x, y, speed, acc, dir, orient, v_surge, v_sway, a_surge, a_sway, j_surge, j_sway]

        # 3. Apply Temporal Pyramids
        print(f"Applying Temporal Pyramids for Stream B ({len(df)} rows)...")
        X_flattened = self._apply_temporal_pyramids(df, STREAM_B_FEATURES, group_cols)

        # 4. Final Cleanup
        # Extract Target and IDs
        y = df["contact"].values.astype(int)
        ids = df["contact_id"].values

        # Reduce memory
        X_flattened = reduce_mem_usage(X_flattened, verbose=False)

        return {"X": X_flattened, "y": y, "ids": ids}
