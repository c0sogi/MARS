import os
import gc
import numpy as np
import pandas as pd
import scipy.ndimage
from library.config import Config
from library.utils import setup_logging, load_parquet, save_parquet, get_hash

# Setup logging
setup_logging()


class FeatureExtractor:
    def __init__(self):
        self.config = Config
        self.logger = pd.io.common.logging.getLogger(__name__)

    def _load_tracking(self, path):
        """
        Loads and preprocesses tracking data.
        Creates windowed acceleration columns for spectral analysis.
        """
        df = pd.read_csv(path)

        # Ensure correct types
        df["nfl_player_id"] = df["nfl_player_id"].astype(int)
        df["step"] = df["step"].astype(int)

        # Sort for windowing
        df = df.sort_values(["game_play", "nfl_player_id", "step"]).reset_index(
            drop=True
        )

        # Create windowed features for acceleration
        # We need history/future for spectral analysis
        window_size = self.config.FEATURES["WINDOW_SIZE_STEPS"]

        # Group object for shifting
        # We only need acceleration windows for the spectral feature
        grp = df.groupby(["game_play", "nfl_player_id"])["acceleration"]

        new_cols = {}
        for i in range(-window_size, window_size + 1):
            # shift(1) is lag (previous), shift(-1) is lead (future)
            # We want t+i. If i is positive (future), we shift by -i.
            col_name = f"acc_step_{i}"
            new_cols[col_name] = grp.shift(-i)

        # Concatenate new columns efficiently
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

        # Fill NaNs in window columns (edges of play) with the instantaneous acceleration
        acc_cols = list(new_cols.keys())
        for col in acc_cols:
            df[col] = df[col].fillna(df["acceleration"])

        return df, acc_cols

    def _solve_quadratic_min_dist(self, df):
        """
        Vectorized estimation of minimum distance in the lookahead window.
        d(t) = || r0 + v*t + 0.5*a*t^2 ||
        """
        # Time steps to sample (0.0 to 1.0s in 0.1s increments)
        t_steps = np.linspace(0, self.config.GATING["WINDOW_SECONDS"], num=11)

        # Extract vectors
        rx = (df["x_position_p1"] - df["x_position_p2"]).values
        ry = (df["y_position_p1"] - df["y_position_p2"]).values

        # Relative velocity
        # We need velocity components. Tracking gives speed/direction/orientation.
        # Convert polar to cartesian.
        # direction is 0-360 degrees, 0 is Y axis (usually in NFL data), 90 is X.
        # Standard NFL tracking: 0 is along Y (short axis), 90 is along X (long axis).
        # However, usually it's best to verify. Assuming standard math convention or NFL convention.
        # NFL Big Data Bowl standard: 0 is Y, 90 is X.
        # vx = speed * sin(deg2rad(direction))
        # vy = speed * cos(deg2rad(direction))

        def get_comp(speed, direction):
            rad = np.radians(direction)
            vx = speed * np.sin(rad)
            vy = speed * np.cos(rad)
            return vx, vy

        vx1, vy1 = get_comp(df["speed_p1"].values, df["direction_p1"].values)
        vx2, vy2 = get_comp(df["speed_p2"].values, df["direction_p2"].values)

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Acceleration is magnitude. We don't have acceleration direction explicitly
        # usually in this dataset (unless derived). 'sa' is signed accel.
        # We will approximate acceleration direction as velocity direction for simplicity
        # or just use the scalar acceleration magnitude for a worst-case bound?
        # The prompt formula implies vector addition.
        # Let's use the velocity direction for acceleration vector approximation.
        ax1, ay1 = get_comp(df["acceleration_p1"].values, df["direction_p1"].values)
        ax2, ay2 = get_comp(df["acceleration_p2"].values, df["direction_p2"].values)

        dax = ax1 - ax2
        day = ay1 - ay2

        # Compute min dist over time steps
        min_dists = np.full(len(df), np.inf)

        for t in t_steps:
            # r(t) components
            rt_x = rx + dvx * t + 0.5 * dax * (t**2)
            rt_y = ry + dvy * t + 0.5 * day * (t**2)
            dist_t = np.sqrt(rt_x**2 + rt_y**2)
            min_dists = np.minimum(min_dists, dist_t)

        return min_dists

    def _compute_transient_spectral_energy(self, df, acc_cols):
        """
        Computes the RMS energy of the high-frequency component of relative acceleration.
        """
        # Extract P1 and P2 window columns
        p1_cols = [c + "_p1" for c in acc_cols]
        p2_cols = [c + "_p2" for c in acc_cols]

        # Convert to numpy
        acc_p1 = df[p1_cols].values
        acc_p2 = df[p2_cols].values

        # Relative acceleration profile
        acc_rel = acc_p1 - acc_p2

        # Apply Gaussian Filter (Low Pass)
        sigma = self.config.FEATURES["SPECTRAL_SIGMA"]
        low_pass = scipy.ndimage.gaussian_filter1d(acc_rel, sigma=sigma, axis=1)

        # High Pass = Original - Low Pass
        high_pass = acc_rel - low_pass

        # Energy = RMS of High Pass
        energy = np.sqrt(np.mean(high_pass**2, axis=1))

        return energy

    def generate_features(
        self, metadata_path, tracking_path, mode="train", load_cached_data=True
    ):
        """
        Main pipeline for feature generation.
        """
        # 1. Cache Check
        cache_key = f"{mode}_features"
        cache_path = self.config.get_cache_path(cache_key)

        # Create a hash of the config to ensure parameters match
        config_hash = get_hash(self.config.FEATURES) + get_hash(self.config.GATING)
        # Append hash to filename to invalidate cache on config change
        base, ext = os.path.splitext(cache_path)
        hashed_cache_path = f"{base}_{config_hash}{ext}"

        if load_cached_data and os.path.exists(hashed_cache_path):
            print(f"Loading cached features from {hashed_cache_path}")
            return load_parquet(hashed_cache_path)

        print(f"Generating features for {mode}...")

        # 2. Load Data
        df_meta = pd.read_csv(metadata_path)
        df_tracking, acc_window_cols = self._load_tracking(tracking_path)

        # 3. Merge Strategy
        # Split Meta into Player-Player and Player-Ground
        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_pg = df_meta[mask_ground].copy()
        df_pp = df_meta[~mask_ground].copy()

        # Ensure ID types
        df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

        # Columns to keep from tracking
        track_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "direction",
            "orientation",
            "acceleration",
            "sa",
        ] + acc_window_cols

        # --- Process Player-Player ---
        if not df_pp.empty:
            # Merge P1
            df_pp = df_pp.merge(
                df_tracking[track_cols],
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id"])
            # Rename P1 cols
            rename_p1 = {
                c: f"{c}_p1"
                for c in track_cols
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_pp = df_pp.rename(columns=rename_p1)

            # Merge P2
            df_pp = df_pp.merge(
                df_tracking[track_cols],
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id"])
            # Rename P2 cols
            rename_p2 = {
                c: f"{c}_p2"
                for c in track_cols
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_pp = df_pp.rename(columns=rename_p2)

        # --- Process Player-Ground ---
        if not df_pg.empty:
            # Merge P1
            df_pg = df_pg.merge(
                df_tracking[track_cols],
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id"])
            rename_p1 = {
                c: f"{c}_p1"
                for c in track_cols
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_pg = df_pg.rename(columns=rename_p1)

            # Fill P2 with zeros/stationary
            p2_cols = [
                c for c in track_cols if c not in ["game_play", "step", "nfl_player_id"]
            ]
            for c in p2_cols:
                df_pg[f"{c}_p2"] = 0.0

            # Set Ground Sentinel for distance later
            # We can't calculate distance yet, but we will override it.

        # Combine
        df_combined = pd.concat([df_pp, df_pg], axis=0).reset_index(drop=True)
        del df_pp, df_pg
        gc.collect()

        # 4. Feature Engineering

        # Distance
        df_combined["distance"] = np.sqrt(
            (df_combined["x_position_p1"] - df_combined["x_position_p2"]) ** 2
            + (df_combined["y_position_p1"] - df_combined["y_position_p2"]) ** 2
        )

        # Apply Ground Sentinel
        # If P2 was G, distance should be sentinel
        if mask_ground.any():
            # Re-identify ground rows in combined
            ground_rows = df_combined["nfl_player_id_2"] == "G"
            df_combined.loc[ground_rows, "distance"] = self.config.GATING[
                "G_DISTANCE_SENTINEL"
            ]

        # Basic Kinematics
        df_combined["speed_diff"] = np.abs(
            df_combined["speed_p1"] - df_combined["speed_p2"]
        )
        df_combined["acc_diff"] = np.abs(
            df_combined["acceleration_p1"] - df_combined["acceleration_p2"]
        )

        # 5. Quadratic Gating
        # Only apply gating if enabled and NOT in test mode (or if we want to filter test too, but usually not)
        # The prompt says "Generate the Full Spectral Feature Set for the Entire Test Set", so NO gating on test.
        if self.config.GATING["ENABLED"] and mode != "test":
            print(
                f"Applying Quadratic Gating (Pre-filter count: {len(df_combined)})..."
            )

            # Calculate min predicted distance
            # For Ground rows, distance is -1, so they should always pass if threshold > -1
            min_dists = self._solve_quadratic_min_dist(df_combined)

            # Force ground rows to pass
            ground_rows = df_combined["nfl_player_id_2"] == "G"
            min_dists[ground_rows] = -999.0

            # Filter
            mask = min_dists < self.config.GATING["DISTANCE_THRESHOLD"]
            df_combined = df_combined[mask].reset_index(drop=True)
            print(f"Gating Complete (Post-filter count: {len(df_combined)})")

        # 6. Spectral Features
        if self.config.FEATURES["USE_SPECTRAL"]:
            print("Computing Transient Spectral Energy...")
            df_combined["spectral_energy"] = self._compute_transient_spectral_energy(
                df_combined, acc_window_cols
            )

        # 7. Cleanup
        # Drop raw window columns
        cols_to_drop = [c + "_p1" for c in acc_window_cols] + [
            c + "_p2" for c in acc_window_cols
        ]
        # Also drop metadata columns not needed for training
        cols_to_drop += self.config.FEATURES["DROP_COLS"]

        # Only drop columns that actually exist
        existing_drop = [c for c in cols_to_drop if c in df_combined.columns]
        df_final = df_combined.drop(columns=existing_drop)

        # Fill any remaining NaNs (e.g. from missing tracking data)
        df_final = df_final.fillna(0)

        # Save to cache
        print(f"Saving features to {hashed_cache_path}...")
        save_parquet(df_final, hashed_cache_path)

        return df_final
