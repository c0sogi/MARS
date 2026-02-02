import os
import logging
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import parameter_aware_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureEngineer:
    """
    Implements the Dual-Basis Time-Domain Anchored-Mining feature engineering pipeline.
    """

    def __init__(self, config=Config):
        self.config = config
        self.window_size = config.WINDOW_SIZE
        self.ground_sentinel = config.GROUND_DISTANCE_SENTINEL
        self.gating_threshold = config.GATING_DISTANCE_THRESHOLD

    def _load_tracking_data(self, path):
        """
        Loads and preprocesses tracking data.
        Computes velocity and acceleration components from speed/direction.
        Computes Jerk via finite differences.
        """
        logging.info(f"Loading tracking data from {path}...")
        df = pd.read_csv(path)

        # Standardize direction (0=North/Y, 90=East/X convention in NFL data usually)
        # Converting to radians. Assuming 0 is Y-axis, increasing clockwise.
        # v_x = speed * sin(theta)
        # v_y = speed * cos(theta)
        df["dir_rad"] = np.radians(df["direction"])

        df["v_x"] = df["speed"] * np.sin(df["dir_rad"])
        df["v_y"] = df["speed"] * np.cos(df["dir_rad"])

        # Sort for finite differences
        df = df.sort_values(["game_play", "nfl_player_id", "step"])

        # Compute Acceleration components (finite diff of velocity)
        # Note: 'acceleration' column is magnitude. We want vector components.
        # We group by player to ensure boundaries are respected.
        # Using shift to calculate diffs.
        # Time step is 0.1s. a = dv/dt = (v_t - v_{t-1}) / 0.1
        grp = df.groupby(["game_play", "nfl_player_id"])

        df["a_x"] = grp["v_x"].diff() / 0.1
        df["a_y"] = grp["v_y"].diff() / 0.1

        # Fill NaNs (first step of each play) with 0 or the first valid value
        df["a_x"] = df["a_x"].fillna(0)
        df["a_y"] = df["a_y"].fillna(0)

        # Compute Jerk components (finite diff of acceleration)
        df["j_x"] = grp["a_x"].diff() / 0.1
        df["j_y"] = grp["a_y"].diff() / 0.1

        df["j_x"] = df["j_x"].fillna(0)
        df["j_y"] = df["j_y"].fillna(0)

        # Select relevant columns
        cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "v_x",
            "v_y",
            "a_x",
            "a_y",
            "j_x",
            "j_y",
        ]
        return df[cols]

    def _merge_tracking_at_lag(self, df_meta, df_track, lag):
        """
        Merges tracking data for P1 and P2 at a specific time lag (step + lag).
        """
        # Create lag step
        df_meta_lag = df_meta.copy()
        df_meta_lag["step_lag"] = df_meta_lag["step"] + lag

        # Merge P1
        df_merged = pd.merge(
            df_meta_lag,
            df_track,
            left_on=["game_play", "step_lag", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1"),
        )
        # Rename P1 columns to be explicit (merge might not add suffix if no collision, but we want explicit)
        track_cols = [
            "x_position",
            "y_position",
            "v_x",
            "v_y",
            "a_x",
            "a_y",
            "j_x",
            "j_y",
        ]
        rename_p1 = {col: f"{col}_p1_lag{lag}" for col in track_cols}

        # Handle columns that might have collided or not
        for col in track_cols:
            if col in df_merged.columns:
                df_merged = df_merged.rename(columns={col: rename_p1[col]})
            elif f"{col}_y" in df_merged.columns:  # if collision happened
                df_merged = df_merged.rename(columns={f"{col}_y": rename_p1[col]})

        # Drop redundant
        df_merged = df_merged.drop(
            columns=["nfl_player_id", "step_lag", "step_y"], errors="ignore"
        )
        if "step_x" in df_merged.columns:
            df_merged = df_merged.rename(columns={"step_x": "step"})

        # Prepare P2 merge
        # Map 'G' to a dummy ID (e.g., -1) or handle separately.
        # We treat 'G' as missing tracking data, which will result in NaNs, which we fill with 0.
        # But we need to join on numeric ID.
        df_merged["join_id_2"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        # Merge P2
        df_merged = pd.merge(
            df_merged,
            df_track,
            left_on=["game_play", "step_lag", "join_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2"),
        )

        rename_p2 = {col: f"{col}_p2_lag{lag}" for col in track_cols}
        for col in track_cols:
            if col in df_merged.columns:
                df_merged = df_merged.rename(columns={col: rename_p2[col]})
            elif f"{col}_y" in df_merged.columns:
                df_merged = df_merged.rename(columns={f"{col}_y": rename_p2[col]})

        # Drop redundant
        df_merged = df_merged.drop(
            columns=["nfl_player_id", "step_lag", "join_id_2", "step_y"],
            errors="ignore",
        )
        if "step_x" in df_merged.columns:
            df_merged = df_merged.rename(columns={"step_x": "step"})

        # Fill NaNs for P2 (Ground or missing data) with 0
        # This effectively sets Ground position/velocity to 0, which is handled by the Basis logic later.
        p2_cols = list(rename_p2.values())
        df_merged[p2_cols] = df_merged[p2_cols].fillna(0.0)

        # Fill NaNs for P1 (edge of play) with 0
        p1_cols = list(rename_p1.values())
        df_merged[p1_cols] = df_merged[p1_cols].fillna(0.0)

        return df_merged

    def _compute_dual_basis_and_project(self, df):
        """
        Implements Stage 1: Dual-Basis Time-Domain Feature Engineering.
        """
        # 1. Define Basis Vectors (u_x, u_y) at lag 0 (Anchor time)
        # Case A: Player-Player -> Collision Axis (P2 -> P1)
        # Case B: Player-Ground -> Motion Axis (P1 Velocity)

        # Extract lag 0 coordinates
        x1 = df["x_position_p1_lag0"]
        y1 = df["y_position_p1_lag0"]
        x2 = df["x_position_p2_lag0"]
        y2 = df["y_position_p2_lag0"]
        vx1 = df["v_x_p1_lag0"]
        vy1 = df["v_y_p1_lag0"]

        is_ground = df["nfl_player_id_2"] == "G"

        # Initialize basis vectors
        u_x = np.zeros(len(df))
        u_y = np.zeros(len(df))

        # --- Case A: Player-Player ---
        # Vector r_12 = p1 - p2
        dx = x1 - x2
        dy = y1 - y2
        dist = np.sqrt(dx**2 + dy**2)
        # Avoid division by zero
        dist_safe = np.where(dist < 1e-6, 1e-6, dist)

        # Basis u = r_12 / |r_12|
        u_x_pp = dx / dist_safe
        u_y_pp = dy / dist_safe

        # --- Case B: Player-Ground ---
        # Basis u = v_1 / |v_1|
        speed = np.sqrt(vx1**2 + vy1**2)
        speed_safe = np.where(speed < 1e-6, 1e-6, speed)
        u_x_pg = vx1 / speed_safe
        u_y_pg = vy1 / speed_safe

        # Assign based on mask
        u_x = np.where(is_ground, u_x_pg, u_x_pp)
        u_y = np.where(is_ground, u_y_pg, u_y_pp)

        # Orthogonal Basis u_perp (Rotate 90 deg: -y, x)
        u_perp_x = -u_y
        u_perp_y = u_x

        # 2. Time-Domain Decomposition for all lags
        # We project relative vectors onto u and u_perp
        # Relative Vector = V_p1 - V_p2 (For Ground, V_p2 is 0, so just V_p1)

        feature_dict = {}

        # Min distance tracker for Gating
        min_distances = np.full(len(df), 999.0)

        lags = range(-self.window_size, self.window_size + 1)
        for lag in lags:
            # Extract raw vectors
            vp1_x = df[f"v_x_p1_lag{lag}"]
            vp1_y = df[f"v_y_p1_lag{lag}"]
            vp2_x = df[f"v_x_p2_lag{lag}"]
            vp2_y = df[f"v_y_p2_lag{lag}"]

            ap1_x = df[f"a_x_p1_lag{lag}"]
            ap1_y = df[f"a_y_p1_lag{lag}"]
            ap2_x = df[f"a_x_p2_lag{lag}"]
            ap2_y = df[f"a_y_p2_lag{lag}"]

            jp1_x = df[f"j_x_p1_lag{lag}"]
            jp1_y = df[f"j_y_p1_lag{lag}"]
            jp2_x = df[f"j_x_p2_lag{lag}"]
            jp2_y = df[f"j_y_p2_lag{lag}"]

            xp1 = df[f"x_position_p1_lag{lag}"]
            yp1 = df[f"y_position_p1_lag{lag}"]
            xp2 = df[f"x_position_p2_lag{lag}"]
            yp2 = df[f"y_position_p2_lag{lag}"]

            # Relative Vectors
            v_rel_x = vp1_x - vp2_x
            v_rel_y = vp1_y - vp2_y
            a_rel_x = ap1_x - ap2_x
            a_rel_y = ap1_y - ap2_y
            j_rel_x = jp1_x - jp2_x
            j_rel_y = jp1_y - jp2_y

            # Project Velocity
            # Comp 1 (Radial/Longitudinal) = v . u
            feature_dict[f"v_comp1_lag{lag}"] = v_rel_x * u_x + v_rel_y * u_y
            # Comp 2 (Tangential/Lateral) = v . u_perp
            feature_dict[f"v_comp2_lag{lag}"] = v_rel_x * u_perp_x + v_rel_y * u_perp_y

            # Project Acceleration
            feature_dict[f"a_comp1_lag{lag}"] = a_rel_x * u_x + a_rel_y * u_y
            feature_dict[f"a_comp2_lag{lag}"] = a_rel_x * u_perp_x + a_rel_y * u_perp_y

            # Project Jerk
            feature_dict[f"j_comp1_lag{lag}"] = j_rel_x * u_x + j_rel_y * u_y
            feature_dict[f"j_comp2_lag{lag}"] = j_rel_x * u_perp_x + j_rel_y * u_perp_y

            # Calculate Distance at this lag
            d_x = xp1 - xp2
            d_y = yp1 - yp2
            dist_lag = np.sqrt(d_x**2 + d_y**2)

            # Update min distance for gating
            # Note: For Ground, dist_lag is distance to (0,0) which is wrong.
            # But Ground rows are handled by sentinel later.
            # For Gating, P-G is always kept, so we only care about P-P distance here.
            min_distances = np.minimum(min_distances, dist_lag)

            # Store distance feature (overridden for Ground later)
            feature_dict[f"dist_lag{lag}"] = dist_lag

        # Create DataFrame from features
        df_feats = pd.DataFrame(feature_dict)

        # Add Sentinel for Ground Distance
        # If Ground, set all distance lags to Sentinel
        for lag in lags:
            df_feats.loc[is_ground, f"dist_lag{lag}"] = self.ground_sentinel

        # Add Physics Primitives
        # Time-To-Collision (TTC) at lag 0
        # TTC = Distance / Closing Speed (Negative Radial Velocity)
        # Closing Speed = -v_comp1_lag0
        dist_0 = df_feats["dist_lag0"]
        v_rad_0 = df_feats["v_comp1_lag0"]
        # Avoid div by zero
        ttc = dist_0 / -(v_rad_0 - 1e-6)
        # Clip TTC to reasonable range (e.g., 0 to 5s), set negative (moving away) to high value
        ttc = np.where((v_rad_0 < 0) & (dist_0 > 0), ttc, 10.0)
        df_feats["ttc"] = ttc

        # Attach Gating Metric
        # For Ground, min_distance calculated above is junk. Force it to 0 so it passes gating.
        final_min_dist = np.where(is_ground, 0.0, min_distances)
        df_feats["min_dist_gating"] = final_min_dist

        return df_feats

    @parameter_aware_cache(Config.CACHE_TRAIN_FEATURES, file_format="parquet")
    def create_train_features(self, load_cached_data=False, debug=False):
        return self._pipeline(
            self.config.TRAIN_METADATA_PATH,
            self.config.TRAIN_TRACKING_PATH,
            is_train=True,
            debug=debug,
        )

    @parameter_aware_cache(Config.CACHE_VAL_FEATURES, file_format="parquet")
    def create_val_features(self, load_cached_data=False, debug=False):
        return self._pipeline(
            self.config.VAL_METADATA_PATH,
            self.config.TRAIN_TRACKING_PATH,
            is_train=True,
            debug=debug,
        )

    @parameter_aware_cache(Config.CACHE_TEST_FEATURES, file_format="parquet")
    def create_test_features(self, load_cached_data=False, debug=False):
        return self._pipeline(
            self.config.TEST_METADATA_PATH,
            self.config.TEST_TRACKING_PATH,
            is_train=False,
            debug=debug,
        )

    def _pipeline(self, metadata_path, tracking_path, is_train=True, debug=False):
        """
        Main execution pipeline.
        """
        # 1. Load Metadata
        logging.info(f"Loading metadata from {metadata_path}")
        df_meta = pd.read_csv(metadata_path)

        if debug:
            logging.info(f"Debug mode: Sampling {self.config.DEBUG_SAMPLE_SIZE} rows.")
            if len(df_meta) > self.config.DEBUG_SAMPLE_SIZE:
                df_meta = df_meta.sample(
                    n=self.config.DEBUG_SAMPLE_SIZE, random_state=self.config.SEED
                ).reset_index(drop=True)

        # 2. Load Tracking
        df_track = self._load_tracking_data(tracking_path)

        # 3. Merge Lags
        logging.info("Merging tracking data across time lags...")
        # Start with metadata
        df_combined = df_meta.copy()

        # Iteratively merge for each lag
        for lag in range(-self.window_size, self.window_size + 1):
            df_combined = self._merge_tracking_at_lag(df_combined, df_track, lag)

        # 4. Compute Features
        logging.info("Computing Dual-Basis Time-Domain features...")
        df_features = self._compute_dual_basis_and_project(df_combined)

        # 5. Attach Identifiers and Target
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        if is_train:
            meta_cols.append("contact")

        df_final = pd.concat([df_meta[meta_cols], df_features], axis=1)

        # 6. Relaxed Quadratic Gating
        logging.info(
            f"Applying Relaxed Quadratic Gating (Threshold: {self.gating_threshold} yards)..."
        )
        initial_len = len(df_final)
        df_final = df_final[df_final["min_dist_gating"] < self.gating_threshold].copy()
        df_final = df_final.drop(columns=["min_dist_gating"])
        logging.info(
            f"Gating reduced dataset from {initial_len} to {len(df_final)} rows."
        )

        return df_final


def generate_features(mode="train", load_cached_data=False, debug=False):
    """
    Wrapper function to be called by the main runner.
    """
    engineer = FeatureEngineer(Config)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if mode == "train":
        return engineer.create_train_features(
            load_cached_data=load_cached_data, debug=debug
        )
    elif mode == "val":
        return engineer.create_val_features(
            load_cached_data=load_cached_data, debug=debug
        )
    elif mode == "test":
        return engineer.create_test_features(
            load_cached_data=load_cached_data, debug=debug
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
