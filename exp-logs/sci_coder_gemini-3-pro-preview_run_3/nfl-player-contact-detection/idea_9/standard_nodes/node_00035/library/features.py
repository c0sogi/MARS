import pandas as pd
import numpy as np
import os
import gc
from typing import Dict, Tuple, List, Optional

from library.config import (
    WORKING_DIR,
    STREAM_CONFIG,
    WINDOW_MICRO,
    WINDOW_MACRO,
    TRACKING_BASE_COLS,
    TRACKING_DERIVED_COLS_A,
    TRACKING_DERIVED_COLS_B,
    VISUAL_BASE_COLS,
    VISUAL_DERIVED_COLS,
    VIEWS,
    SEED,
)
from library.utils import generate_config_hash
from library.data_factory import load_dataset, partition_streams


class FeatureEngineer:
    """
    Implements the Asymmetric Modality-Selective feature engineering pipeline.
    Handles data loading, preprocessing, windowing, and stream-specific feature generation
    with caching support.
    """

    def __init__(self, mode: str = "train"):
        """
        Args:
            mode: 'train', 'validation', or 'test'.
        """
        self.mode = mode
        self.labels_df = None
        self.tracking_df = None
        self.helmets_df = None

        # Define the configuration dictionary for hashing
        self.config_dict = {
            "window_micro": WINDOW_MICRO,
            "window_macro": WINDOW_MACRO,
            "stream_config": STREAM_CONFIG,
            "tracking_base": TRACKING_BASE_COLS,
            "visual_base": VISUAL_BASE_COLS,
        }
        self.config_hash = generate_config_hash(self.config_dict)

    def _get_cache_path(self, stream_name: str, file_type: str) -> str:
        """Generates a file path for caching based on mode, stream, and config hash."""
        filename = f"{self.mode}_{stream_name}_{self.config_hash}_{file_type}"
        if file_type == "X":
            filename += ".parquet"
        else:
            filename += ".npy"
        return os.path.join(WORKING_DIR, filename)

    def _preprocess_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares tracking data: sorts, computes derivatives, and generates windowed features
        on the full dataset before merging to save computation.
        """
        # Sort for windowing
        df = df.sort_values(["game_play", "nfl_player_id", "step"]).reset_index(
            drop=True
        )

        # --- Compute Derivatives needed for Stream B (Jerk, Alignment) ---
        # Note: We compute these for all, but only select them later for Stream B

        # Pose-Motion Alignment: Dot product of orientation and direction
        # Orientation and Direction are in degrees. Convert to radians.
        # 0 degrees is usually Y-axis or X-axis depending on convention, but relative diff matters.
        # We assume standard unit circle or consistent offset.
        # Alignment = cos(direction - orientation)

        # Fill NaNs in angles
        df["direction"] = df["direction"].fillna(0)
        df["orientation"] = df["orientation"].fillna(0)

        rad_dir = np.radians(df["direction"])
        rad_orient = np.radians(df["orientation"])
        df["pose_motion_alignment"] = np.cos(rad_dir - rad_orient)

        # Jerk: Derivative of acceleration
        # Group by play/player to prevent boundary bleeding
        # acceleration is scalar magnitude.
        df["jerk"] = (
            df.groupby(["game_play", "nfl_player_id"])["acceleration"].diff().fillna(0)
        )

        # --- Generate Micro Window Features (Lags) ---
        # We create lag columns for ALL base features + derived ones
        # Features to lag: Base + Jerk + Alignment
        cols_to_lag = TRACKING_BASE_COLS + ["jerk", "pose_motion_alignment"]

        # We will store the names of generated columns to retrieve them easily
        lag_cols = []

        # Use groupby shift for safety across game_plays
        grouper = df.groupby(["game_play", "nfl_player_id"])

        for lag in range(-WINDOW_MICRO, WINDOW_MICRO + 1):
            suffix = f"_lag{lag}"
            if lag == 0:
                # Just rename current columns to match pattern or keep as is?
                # Let's rename to _lag0 for consistency in downstream processing
                for col in cols_to_lag:
                    df[f"{col}{suffix}"] = df[col]
                    lag_cols.append(f"{col}{suffix}")
            else:
                # Shift
                shifted = grouper[cols_to_lag].shift(
                    -lag
                )  # positive lag means future in shift(-k)?
                # shift(1) moves t to t+1 (gets previous). shift(-1) gets next.
                # We want t-4 (past) to t+4 (future).
                # lag -4: data from 4 steps ago. shift(4).
                # lag +4: data from 4 steps future. shift(-4).
                shifted = grouper[cols_to_lag].shift(
                    lag
                )  # shift(k) takes value from t-k and puts it at t

                # Rename
                shifted.columns = [f"{c}{suffix}" for c in cols_to_lag]

                # Concatenate (efficiently)
                df = pd.concat([df, shifted], axis=1)
                lag_cols.extend(shifted.columns)

        # --- Generate Macro Window Features (Rolling Aggregates) ---
        # Window size = 2 * WINDOW_MACRO + 1
        window_size = 2 * WINDOW_MACRO + 1

        # Rolling stats on specific columns (Speed, Acceleration)
        cols_to_roll = ["speed", "acceleration", "sa"]
        roll_stats = ["mean", "std", "max"]

        for col in cols_to_roll:
            rolled = grouper[col].rolling(
                window=window_size, center=True, min_periods=1
            )
            for stat in roll_stats:
                col_name = f"{col}_roll_{stat}"
                if stat == "mean":
                    df[col_name] = (
                        rolled.mean()
                        .reset_index(0, drop=True)
                        .reset_index(0, drop=True)
                    )
                elif stat == "std":
                    df[col_name] = (
                        rolled.std().reset_index(0, drop=True).reset_index(0, drop=True)
                    )
                elif stat == "max":
                    df[col_name] = (
                        rolled.max().reset_index(0, drop=True).reset_index(0, drop=True)
                    )

        # Fill NaNs generated by shifts/rolling
        # For tracking, 0 is often a safe default for speed/accel, but position needs care.
        # However, we mostly use relative positions later.
        # We'll fill numeric columns with 0 for simplicity in this pipeline
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)

        return df

    def _preprocess_helmets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares helmet data: Maps frames to steps and generates windowed features.
        """
        # Map frame to step.
        # Step 0 = Frame 300 (approx). 1 step = 6 frames.
        # We want the helmet box closest to the step time.
        # frame = 300 + step * 6
        # Inverse: step = (frame - 300) / 6
        # We filter to frames that are multiples of 6 offset by 300 (or close enough)

        # Filter roughly to relevant frames to reduce size
        # We only care about integer steps.
        # Let's calculate 'estimated_step' and round.
        df["est_step"] = (df["frame"] - 300) / 6.0
        df["step"] = df["est_step"].round()

        # Filter to where the frame is close to the exact step (within 1 frame)
        # 59.94Hz vs 60Hz drift is small within a play.
        df = df[np.abs(df["est_step"] - df["step"]) < 0.2].copy()
        df["step"] = df["step"].astype(int)

        # Sort
        df = df.sort_values(["game_play", "view", "nfl_player_id", "step"]).reset_index(
            drop=True
        )

        # Generate Micro Lags for BBox columns
        cols_to_lag = VISUAL_BASE_COLS  # left, width, top, height
        grouper = df.groupby(["game_play", "view", "nfl_player_id"])

        for lag in range(-WINDOW_MICRO, WINDOW_MICRO + 1):
            suffix = f"_lag{lag}"
            if lag == 0:
                for col in cols_to_lag:
                    df[f"{col}{suffix}"] = df[col]
            else:
                shifted = grouper[cols_to_lag].shift(lag)
                shifted.columns = [f"{c}{suffix}" for c in cols_to_lag]
                df = pd.concat([df, shifted], axis=1)

        # Fill NaNs (missing boxes) with -1 or 0
        df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(
            include=[np.number]
        ).fillna(0)

        return df

    def _compute_interaction_features(
        self, df: pd.DataFrame, lag_suffix: str
    ) -> pd.DataFrame:
        """
        Computes interaction features (Distance, Relative Speed) for a specific lag.
        Operates on columns like x_position_p1_lag0, x_position_p2_lag0.
        """
        # Distance
        dx = df[f"x_position_p1{lag_suffix}"] - df[f"x_position_p2{lag_suffix}"]
        dy = df[f"y_position_p1{lag_suffix}"] - df[f"y_position_p2{lag_suffix}"]
        dist_col = f"distance{lag_suffix}"
        df[dist_col] = np.sqrt(dx**2 + dy**2)

        # Relative Speed (Scalar difference)
        # Note: True relative velocity is vector diff magnitude, but prompt suggests "Relative Speed"
        # We'll use absolute difference of scalar speeds as a simple proxy,
        # plus closure rate handles the vector component implicitly via distance change.
        s1 = df[f"speed_p1{lag_suffix}"]
        s2 = df[f"speed_p2{lag_suffix}"]
        df[f"relative_speed{lag_suffix}"] = np.abs(s1 - s2)

        return df

    def _compute_visual_interaction(
        self, df: pd.DataFrame, view: str, lag_suffix: str
    ) -> pd.DataFrame:
        """
        Computes IoU and Pixel Distance for a specific view and lag.
        """
        # Columns: left_p1_Sideline_lag0, ...
        p1_prefix = f"p1_{view}"
        p2_prefix = f"p2_{view}"

        l1 = df[f"left_{p1_prefix}{lag_suffix}"]
        t1 = df[f"top_{p1_prefix}{lag_suffix}"]
        w1 = df[f"width_{p1_prefix}{lag_suffix}"]
        h1 = df[f"height_{p1_prefix}{lag_suffix}"]
        r1 = l1 + w1
        b1 = t1 + h1

        l2 = df[f"left_{p2_prefix}{lag_suffix}"]
        t2 = df[f"top_{p2_prefix}{lag_suffix}"]
        w2 = df[f"width_{p2_prefix}{lag_suffix}"]
        h2 = df[f"height_{p2_prefix}{lag_suffix}"]
        r2 = l2 + w2
        b2 = t2 + h2

        # Intersection
        x_left = np.maximum(l1, l2)
        y_top = np.maximum(t1, t2)
        x_right = np.minimum(r1, r2)
        y_bottom = np.minimum(b1, b2)

        intersection_area = np.maximum(0, x_right - x_left) * np.maximum(
            0, y_bottom - y_top
        )

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        # IoU
        iou_col = f"iou_{view}{lag_suffix}"
        # Avoid div by zero
        df[iou_col] = intersection_area / (union_area + 1e-6)

        # Centroid Distance
        c1_x, c1_y = l1 + w1 / 2, t1 + h1 / 2
        c2_x, c2_y = l2 + w2 / 2, t2 + h2 / 2

        dist_col = f"dist_pixel_{view}{lag_suffix}"
        df[dist_col] = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

        # Normalize pixel distance by image width (approx 1280) to keep scale reasonable
        df[dist_col] = df[dist_col] / 1280.0

        return df

    def _generate_stream_a(
        self, df_labels: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generates features for Stream A (Player-Player).
        Fusion of Tracking (P1 & P2) + Visual (P1 & P2).
        """
        print("Generating Stream A (Player-Player) features...")

        # Ensure join keys are numeric to match tracking data (Cite debug_lesson_3)
        df_labels["nfl_player_id_1"] = pd.to_numeric(
            df_labels["nfl_player_id_1"], errors="coerce"
        )
        df_labels["nfl_player_id_2"] = pd.to_numeric(
            df_labels["nfl_player_id_2"], errors="coerce"
        )

        # 1. Merge P1 Tracking
        # We need to join on game_play, step, nfl_player_id
        # Tracking df is already windowed.
        track_cols = [
            c
            for c in self.tracking_df.columns
            if c
            not in [
                "game_play",
                "step",
                "nfl_player_id",
                "datetime",
                "position",
                "team",
                "jersey_number",
            ]
        ]

        # Rename tracking cols for P1
        p1_track = self.tracking_df[
            ["game_play", "step", "nfl_player_id"] + track_cols
        ].copy()

        # Correctly rename columns: Insert _p1 before _lag suffix if present
        new_cols_p1 = ["game_play", "step", "nfl_player_id_1"]
        for c in track_cols:
            if "_lag" in c:
                parts = c.split("_lag")
                # e.g., x_position_lag-4 -> x_position_p1_lag-4
                new_cols_p1.append(f"{parts[0]}_p1_lag{parts[1]}")
            else:
                new_cols_p1.append(f"{c}_p1")
        p1_track.columns = new_cols_p1

        df_merged = pd.merge(
            df_labels, p1_track, on=["game_play", "step", "nfl_player_id_1"], how="left"
        )

        # 2. Merge P2 Tracking
        p2_track = self.tracking_df[
            ["game_play", "step", "nfl_player_id"] + track_cols
        ].copy()

        # Correctly rename columns: Insert _p2 before _lag suffix if present
        new_cols_p2 = ["game_play", "step", "nfl_player_id_2"]
        for c in track_cols:
            if "_lag" in c:
                parts = c.split("_lag")
                new_cols_p2.append(f"{parts[0]}_p2_lag{parts[1]}")
            else:
                new_cols_p2.append(f"{c}_p2")
        p2_track.columns = new_cols_p2

        df_merged = pd.merge(
            df_merged, p2_track, on=["game_play", "step", "nfl_player_id_2"], how="left"
        )

        # 3. Compute Tracking Interaction Features (Distance, etc.) for each Lag
        # Also compute Closure Rate: -(dist_t - dist_t-1)
        # We need dist at lag t and lag t-1.

        # First, compute distance/rel_speed for all lags
        for lag in range(-WINDOW_MICRO, WINDOW_MICRO + 1):
            suffix = f"_lag{lag}"
            df_merged = self._compute_interaction_features(df_merged, suffix)

        # Now compute Closure Rate (using lag0 and lag-1, lag1 and lag0, etc.)
        # Actually, closure rate is usually instantaneous. Let's compute it for lag 0 using lag -1.
        # Closure Rate t = Distance(t-1) - Distance(t). (Positive if closing in).
        # We can compute this for every lag window if we want, but let's stick to the prompt:
        # "Flattened... Closure Rate". Implies closure rate at each step t?
        # Let's compute closure rate for each lag i using dist_lag_i and dist_lag_(i-1).
        # Note: We computed lags -4 to +4. We can compute closure for -3 to +4.
        # For lag -4, we'd need lag -5.
        # To simplify, we'll just compute closure rate for the available pairs.
        for lag in range(-WINDOW_MICRO + 1, WINDOW_MICRO + 1):
            curr = f"_lag{lag}"
            prev = f"_lag{lag-1}"
            df_merged[f"closure_rate{curr}"] = (
                df_merged[f"distance{prev}"] - df_merged[f"distance{curr}"]
            )

        # Fill missing closure for the first lag with 0
        df_merged[f"closure_rate_lag{-WINDOW_MICRO}"] = 0

        # 4. Merge Visual Data (P1 & P2, Sideline & Endzone)
        # Helmets df has columns: game_play, view, nfl_player_id, step + lag cols
        visual_cols = [c for c in self.helmets_df.columns if "lag" in c]

        for view in VIEWS:
            # Filter helmets by view
            h_view = self.helmets_df[self.helmets_df["view"] == view].copy()

            # Strictly select columns to avoid schema collisions (Cite debug_lesson_11)
            # We need join keys + feature columns (lags)
            cols_to_keep = ["game_play", "step", "nfl_player_id"] + visual_cols
            h_view = h_view[cols_to_keep].copy()

            # Re-do Merge P1
            rename_map_p1 = {}
            rename_map_p2 = {}
            for c in visual_cols:
                # c = 'left_lag-4'
                parts = c.split("_lag")
                base = parts[0]  # 'left'
                lag_part = "_lag" + parts[1]  # '_lag-4'

                rename_map_p1[c] = f"{base}_p1_{view}{lag_part}"
                rename_map_p2[c] = f"{base}_p2_{view}{lag_part}"

            p1_h = h_view.rename(columns=rename_map_p1)
            # p1_h cols: game_play, nfl_player_id, step, left_p1_Sideline_lag-4...
            p1_h = p1_h.rename(columns={"nfl_player_id": "nfl_player_id_1"})

            df_merged = pd.merge(
                df_merged, p1_h, on=["game_play", "step", "nfl_player_id_1"], how="left"
            )

            p2_h = h_view.rename(columns=rename_map_p2)
            p2_h = p2_h.rename(columns={"nfl_player_id": "nfl_player_id_2"})

            df_merged = pd.merge(
                df_merged, p2_h, on=["game_play", "step", "nfl_player_id_2"], how="left"
            )

            # Compute IoU/Dist
            for lag in range(-WINDOW_MICRO, WINDOW_MICRO + 1):
                suffix = f"_lag{lag}"
                df_merged = self._compute_visual_interaction(df_merged, view, suffix)

        # 5. Final Selection
        # Fill NaNs (missing visual data or tracking)
        df_merged = df_merged.fillna(0)

        # Identify feature columns
        # All columns that are not metadata
        meta_cols = [
            "contact_id",
            "game_play",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
        ]
        feature_cols = [c for c in df_merged.columns if c not in meta_cols]

        # Sort columns to ensure deterministic order
        feature_cols.sort()

        X = df_merged[feature_cols].copy()
        y = df_merged["contact"].values.astype(int)
        ids = df_merged["contact_id"].values

        return X, y, ids

    def _generate_stream_b(
        self, df_labels: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generates features for Stream B (Player-Ground).
        Tracking Only (P1). Includes Jerk and Alignment. No Visuals. No P2.
        """
        print("Generating Stream B (Player-Ground) features...")

        # 1. Merge P1 Tracking
        # We need Jerk and Alignment which are in the preprocessed tracking
        track_cols = [
            c
            for c in self.tracking_df.columns
            if c
            not in [
                "game_play",
                "step",
                "nfl_player_id",
                "datetime",
                "position",
                "team",
                "jersey_number",
            ]
        ]

        # Rename tracking cols for P1
        p1_track = self.tracking_df[
            ["game_play", "step", "nfl_player_id"] + track_cols
        ].copy()
        # For Stream B, we don't necessarily need the _p1 suffix since there is no p2,
        # but to keep feature names clean/consistent, we can use it or just keep raw names.
        # Let's use _p1 to indicate the subject player.
        p1_track.columns = ["game_play", "step", "nfl_player_id_1"] + [
            f"{c}_p1" for c in track_cols
        ]

        df_merged = pd.merge(
            df_labels, p1_track, on=["game_play", "step", "nfl_player_id_1"], how="left"
        )

        # 2. No P2 Tracking (Ground)
        # 3. No Visuals

        # 4. Final Selection
        df_merged = df_merged.fillna(0)

        meta_cols = [
            "contact_id",
            "game_play",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
        ]
        feature_cols = [c for c in df_merged.columns if c not in meta_cols]

        feature_cols.sort()

        X = df_merged[feature_cols].copy()
        y = df_merged["contact"].values.astype(int)
        ids = df_merged["contact_id"].values

        return X, y, ids

    def generate_features(
        self, load_cached_data: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
        """
        Main method to generate features for both streams.
        Returns a dictionary: {'A': (X, y, ids), 'B': (X, y, ids)}
        """
        # 1. Check Cache for both streams
        streams = ["A", "B"]
        results = {}
        missing_cache = False

        for stream in streams:
            x_path = self._get_cache_path(stream, "X")
            y_path = self._get_cache_path(stream, "y")
            ids_path = self._get_cache_path(stream, "ids")

            if (
                load_cached_data
                and os.path.exists(x_path)
                and os.path.exists(y_path)
                and os.path.exists(ids_path)
            ):
                print(f"Loading cached features for Stream {stream}...")
                X = pd.read_parquet(x_path)
                y = np.load(y_path)
                ids = np.load(ids_path)
                results[stream] = (X, y, ids)
            else:
                missing_cache = True

        if not missing_cache:
            return results

        # 2. If cache missing, load raw data and compute
        print("Computing features from scratch...")
        self.labels_df, self.tracking_df, self.helmets_df = load_dataset(
            self.mode, load_cached_data
        )

        # Preprocess shared data
        print("Preprocessing Tracking...")
        self.tracking_df = self._preprocess_tracking(self.tracking_df)

        if self.mode != "test":
            # Helmets only needed for Stream A (Player-Player)
            # But we load them anyway.
            pass
        print("Preprocessing Helmets...")
        self.helmets_df = self._preprocess_helmets(self.helmets_df)

        # Partition Labels
        df_a, df_b = partition_streams(self.labels_df)

        # Generate Stream A
        if "A" not in results:
            X_a, y_a, ids_a = self._generate_stream_a(df_a)
            # Save Cache
            X_a.to_parquet(self._get_cache_path("A", "X"), index=False)
            np.save(self._get_cache_path("A", "y"), y_a)
            np.save(self._get_cache_path("A", "ids"), ids_a)
            results["A"] = (X_a, y_a, ids_a)

            # Clean up to save memory
            del X_a, y_a, ids_a, df_a
            gc.collect()

        # Generate Stream B
        if "B" not in results:
            X_b, y_b, ids_b = self._generate_stream_b(df_b)
            # Save Cache
            X_b.to_parquet(self._get_cache_path("B", "X"), index=False)
            np.save(self._get_cache_path("B", "y"), y_b)
            np.save(self._get_cache_path("B", "ids"), ids_b)
            results["B"] = (X_b, y_b, ids_b)

            del X_b, y_b, ids_b, df_b
            gc.collect()

        return results
