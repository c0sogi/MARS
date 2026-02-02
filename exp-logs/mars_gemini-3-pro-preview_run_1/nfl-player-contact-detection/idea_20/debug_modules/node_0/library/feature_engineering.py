import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    SENTINEL_VALUE,
    WINDOW_SIZE,
    IDEA_DIR,
    SEED,
)
from library.utils import CacheManager, setup_logger


class FeatureEngineer:
    """
    Implements Spectral-Kinematic Feature Engineering for the RKS-MTE strategy.
    Transforms gated player pairs into flattened temporal windows with physics-based features.
    """

    def __init__(self):
        self.logger = setup_logger("feature_engineer")
        self.cache_manager = CacheManager(cache_dir=IDEA_DIR)
        # Define window offsets: -10 to +10
        self.window_steps = list(range(-WINDOW_SIZE, WINDOW_SIZE + 1))
        self.window_size = len(self.window_steps)

    def _load_tracking(self, split):
        """Loads and types the tracking data."""
        path = TRAIN_TRACKING_PATH if split in ["train", "val"] else TEST_TRACKING_PATH
        self.logger.info(f"Loading tracking data from {path}")
        df = pd.read_csv(path)
        df["game_play"] = df["game_play"].astype(str)
        df["step"] = df["step"].astype(int)
        return df

    def _compute_spectral_energy(self, rel_accel_matrix):
        """
        Computes Transient Spectral Energy (RMS of High-Frequency Acceleration).
        Decomposes signal into Trend (Low-Freq) and Shock (High-Freq).

        Args:
            rel_accel_matrix (np.ndarray): Shape (N_samples, Window_Size)

        Returns:
            np.ndarray: Shape (N_samples,) containing energy values.
        """
        # 1. Compute Trend (Low-Pass Filter)
        # Simple 5-step moving average.
        # We use a valid convolution approach manually for vectorization
        window_len = 5
        kernel = np.ones(window_len) / window_len

        # Pad the matrix edge to handle boundaries for 'same' size output
        # We replicate the edge values
        pad_width = window_len // 2
        padded = np.pad(rel_accel_matrix, ((0, 0), (pad_width, pad_width)), mode="edge")

        # Apply convolution along axis 1
        # Since we can't easily use scipy.signal in restricted env, we use a strided rolling mean trick
        # or a simple loop since window_len is small.
        trend = np.zeros_like(rel_accel_matrix)
        for i in range(rel_accel_matrix.shape[1]):
            # Extract slice from padded array corresponding to window centered at i
            # padded index: i + pad_width. Window is [i : i + window_len]
            slice_win = padded[:, i : i + window_len]
            trend[:, i] = np.mean(slice_win, axis=1)

        # 2. Compute Shock (High-Pass Filter)
        shock = rel_accel_matrix - trend

        # 3. Compute Energy (RMS of Shock)
        energy = np.sqrt(np.mean(shock**2, axis=1))

        return energy

    def create_features(self, df_gated, split="train", load_cached_data=True):
        """
        Main pipeline to generate flattened spectral-kinematic features.

        Args:
            df_gated (pd.DataFrame): The filtered survivors from the gating stage.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use caching.

        Returns:
            pd.DataFrame: Dataframe with flattened features and targets.
        """
        # 1. Cache Check
        cache_params = {
            "split": split,
            "window": WINDOW_SIZE,
            "input_len": len(df_gated),
            "stage": "features_full_spectral",
        }

        if load_cached_data:
            cached_df = self.cache_manager.load(f"features_{split}", cache_params)
            if cached_df is not None:
                self.logger.info(
                    f"Loaded features from cache. Shape: {cached_df.shape}"
                )
                return cached_df

        self.logger.info(
            f"Generating features for {split} set (Input: {len(df_gated)} rows)..."
        )

        if len(df_gated) == 0:
            self.logger.warning(
                "Input gated dataframe is empty. Returning empty feature set."
            )
            return pd.DataFrame()

        # 2. Load Tracking Data
        df_track = self._load_tracking(split)

        # Optimization: Filter tracking to only relevant plays
        relevant_plays = df_gated["game_play"].unique()
        df_track = df_track[df_track["game_play"].isin(relevant_plays)].copy()

        # Select columns (Strictly excluding context/other players)
        track_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
        ]
        df_track = df_track[track_cols]

        # 3. Vectorized Window Expansion
        self.logger.info("Expanding temporal windows...")

        # Ensure contact_id exists
        if "contact_id" not in df_gated.columns:
            # Construct if missing (should be present from metadata)
            df_gated["contact_id"] = (
                df_gated["game_play"]
                + "_"
                + df_gated["step"].astype(str)
                + "_"
                + df_gated["nfl_player_id_1"].astype(str)
                + "_"
                + df_gated["nfl_player_id_2"].astype(str)
            )

        # Prepare base dataframe for expansion
        # We need to preserve metadata and target
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        if "contact" in df_gated.columns:
            meta_cols.append("contact")

        df_base = df_gated[meta_cols].copy()

        # Repeat rows for each window step
        n_rows = len(df_base)
        n_offsets = len(self.window_steps)
        offsets_arr = np.array(self.window_steps)

        # Repeat indices
        df_expanded = df_base.loc[df_base.index.repeat(n_offsets)].reset_index(
            drop=True
        )

        # Tile offsets and calculate actual step
        df_expanded["offset"] = np.tile(offsets_arr, n_rows)
        df_expanded["step_actual"] = df_expanded["step"] + df_expanded["offset"]

        # 4. Merge Tracking Data (P1 & P2)
        self.logger.info("Merging tracking data for windows...")

        # Merge P1
        df_expanded = pd.merge(
            df_expanded,
            df_track,
            left_on=["game_play", "step_actual", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1"),
        )
        rename_p1 = {
            c: f"{c}_p1"
            for c in ["x_position", "y_position", "speed", "acceleration", "direction"]
        }
        df_expanded = df_expanded.rename(columns=rename_p1)
        df_expanded = df_expanded.drop(
            columns=["nfl_player_id", "step_y"], errors="ignore"
        )

        # Merge P2 (Handle Ground)
        df_expanded["nfl_player_id_2_join"] = pd.to_numeric(
            df_expanded["nfl_player_id_2"], errors="coerce"
        )

        df_expanded = pd.merge(
            df_expanded,
            df_track,
            left_on=["game_play", "step_actual", "nfl_player_id_2_join"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2"),
        )
        rename_p2 = {
            c: f"{c}_p2"
            for c in ["x_position", "y_position", "speed", "acceleration", "direction"]
        }
        df_expanded = df_expanded.rename(columns=rename_p2)
        df_expanded = df_expanded.drop(
            columns=["nfl_player_id", "nfl_player_id_2_join", "step_y"], errors="ignore"
        )

        # Clean up memory
        del df_track
        gc.collect()

        # 5. Compute Kinematics
        self.logger.info("Computing physics primitives...")

        is_ground = df_expanded["nfl_player_id_2"] == "G"

        # Distance
        dx = df_expanded["x_position_p1"] - df_expanded["x_position_p2"]
        dy = df_expanded["y_position_p1"] - df_expanded["y_position_p2"]
        df_expanded["distance"] = np.sqrt(dx**2 + dy**2)

        # Sentinel Value Strategy for Ground
        df_expanded.loc[is_ground, "distance"] = SENTINEL_VALUE

        # Helper for vector components
        def get_components(speed, direction):
            rad = np.radians(direction.fillna(0))
            vx = speed * np.cos(rad)
            vy = speed * np.sin(rad)
            return vx, vy

        # Relative Speed
        vx1, vy1 = get_components(df_expanded["speed_p1"], df_expanded["direction_p1"])
        vx2, vy2 = get_components(df_expanded["speed_p2"], df_expanded["direction_p2"])

        # Zero out P2 motion if Ground
        vx2 = np.where(is_ground, 0, vx2)
        vy2 = np.where(is_ground, 0, vy2)

        df_expanded["rel_speed"] = np.sqrt((vx1 - vx2) ** 2 + (vy1 - vy2) ** 2)

        # Relative Acceleration
        # We approximate accel vector using direction
        ax1, ay1 = get_components(
            df_expanded["acceleration_p1"], df_expanded["direction_p1"]
        )
        ax2, ay2 = get_components(
            df_expanded["acceleration_p2"], df_expanded["direction_p2"]
        )

        ax2 = np.where(is_ground, 0, ax2)
        ay2 = np.where(is_ground, 0, ay2)

        df_expanded["rel_accel"] = np.sqrt((ax1 - ax2) ** 2 + (ay1 - ay2) ** 2)

        # 6. Spectral Feature Extraction & Flattening
        self.logger.info("Extracting spectral features and flattening...")

        # Sort to ensure consistent reshaping
        df_expanded = df_expanded.sort_values(by=["contact_id", "offset"])

        # Reshape to (N_samples, Window_Size)
        # Handle NaNs with fill (e.g., missing tracking steps)
        # For distance, we use SENTINEL_VALUE if missing? Or a large value?
        # If tracking is missing, distance is NaN. Let's use 100 yards (safe large val)
        # For speed/accel, use 0.

        # Features to flatten
        feats_map = {"distance": 100.0, "rel_speed": 0.0, "rel_accel": 0.0}

        flat_data = {}

        # Compute Spectral Energy on 'rel_accel'
        rel_accel_mat = (
            df_expanded["rel_accel"].fillna(0.0).values.reshape(-1, n_offsets)
        )
        spectral_energy = self._compute_spectral_energy(rel_accel_mat)
        flat_data["spectral_energy"] = spectral_energy

        # Flatten raw features
        for feat, fill_val in feats_map.items():
            mat = df_expanded[feat].fillna(fill_val).values.reshape(-1, n_offsets)

            for i, off in enumerate(self.window_steps):
                col_name = f"{feat}_{off}"
                flat_data[col_name] = mat[:, i]

        # 7. Construct Final DataFrame
        df_features = pd.DataFrame(flat_data)

        # Attach Metadata (Take every Nth row)
        df_meta_reduced = df_expanded.iloc[::n_offsets][meta_cols].reset_index(
            drop=True
        )

        # Concatenate
        df_final = pd.concat([df_meta_reduced, df_features], axis=1)

        # 8. Save
        self.cache_manager.save(df_final, f"features_{split}", cache_params)

        self.logger.info(f"Feature generation complete. Shape: {df_final.shape}")
        return df_final
