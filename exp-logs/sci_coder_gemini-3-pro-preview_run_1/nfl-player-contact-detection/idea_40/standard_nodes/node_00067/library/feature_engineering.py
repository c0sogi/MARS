import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import reduce_mem_usage


class FeatureEngineer:
    """
    Implements the Relative-Velocity-Aligned Anchored-Mining feature engineering pipeline.
    """

    def __init__(self, config=Config):
        self.config = config

    def process_train(self, load_cached_data=True):
        return self._process_pipeline(
            self.config.TRAIN_METADATA_PATH,
            self.config.TRAIN_TRACKING_PATH,
            "train",
            load_cached_data,
        )

    def process_val(self, load_cached_data=True):
        return self._process_pipeline(
            self.config.VAL_METADATA_PATH,
            self.config.TRAIN_TRACKING_PATH,  # Validation uses train tracking file
            "val",
            load_cached_data,
        )

    def process_test(self, load_cached_data=True):
        return self._process_pipeline(
            self.config.TEST_METADATA_PATH,
            self.config.TEST_TRACKING_PATH,
            "test",
            load_cached_data,
        )

    def _process_pipeline(self, metadata_path, tracking_path, mode, load_cached):
        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(self.config.WORKING_DIR, f"features_{mode}.parquet")

        if load_cached and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating features for {mode}...")

        # 1. Load Data
        df_meta = pd.read_csv(metadata_path)
        if self.config.DEBUG:
            print(f"DEBUG Mode: Sampling {self.config.DEBUG_SAMPLE_SIZE} rows.")

            # Cite debug_lesson_12: Enforce Minority Class Representation in Downsampled Debug Datasets
            if "contact" in df_meta.columns and df_meta["contact"].sum() > 0:
                positives = df_meta[df_meta["contact"] == 1]
                negatives = df_meta[df_meta["contact"] == 0]

                n_total = self.config.DEBUG_SAMPLE_SIZE
                # Ensure we get enough positives for training
                n_pos = min(len(positives), int(n_total * 0.5))
                n_pos = max(n_pos, min(len(positives), 10))  # Try to get at least 10
                n_neg = n_total - n_pos

                df_pos = positives.sample(n=n_pos, random_state=self.config.SEED)
                df_neg = negatives.sample(n=n_neg, random_state=self.config.SEED)

                df_meta = pd.concat([df_pos, df_neg]).sort_index()
                print(
                    f"DEBUG Stratified Sample: {len(df_pos)} Positives, {len(df_neg)} Negatives."
                )
            else:
                df_meta = df_meta.head(self.config.DEBUG_SAMPLE_SIZE)

        df_tracking = pd.read_csv(tracking_path)

        # 2. Merge Tracking
        # This attaches P1 and P2 tracking data to the metadata
        df = self._merge_tracking_data(df_meta, df_tracking)
        del df_meta, df_tracking
        gc.collect()

        # 3. Compute Instantaneous Features (Relative Velocity Alignment)
        df = self._compute_geometric_features(df)

        # 4. Gating Calculation
        # We calculate the mask but do not drop yet, as we need contiguous rows for lags
        gating_mask = self._calculate_gating_mask(df)
        df["gating_active"] = gating_mask.astype(np.int8)

        # 5. Lag Generation
        # Flattens the time-window into a single row
        df = self._create_lagged_features(df)

        # 6. Apply Gating
        # For Train/Val, we drop non-survivors to clean the dataset.
        # For Test, we MUST keep all rows to match sample_submission,
        # but 'gating_active' allows the model to force 0s.
        if mode != "test":
            df = df[df["gating_active"] == 1].reset_index(drop=True)
            # Drop the gating column as it's all 1s now
            df = df.drop(columns=["gating_active"])

        # 7. Cleanup & Save
        df = reduce_mem_usage(df)
        df.to_parquet(cache_path)
        print(f"Features saved to {cache_path}")

        return df

    def _merge_tracking_data(self, df_meta, df_track):
        """
        Merges tracking data for Player 1 and Player 2.
        Handles Ground (P2='G') by filling P2 columns with 0s.
        """
        # Ensure join keys are consistent strings
        df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(str)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)

        # Drop metadata/non-numeric columns from tracking to prevent merge collisions and schema pollution
        # Cite debug_lesson_15: Verify Target Column Existence Before Renaming to Avoid Merge Suffix Collisions
        # Cite debug_lesson_19: Enforce Explicit Type Filtering for Model Features
        cols_to_drop = ["datetime", "game_key", "play_id", "position", "team"]
        df_track = df_track.drop(columns=cols_to_drop, errors="ignore")

        # Merge Player 1
        df = pd.merge(
            df_meta,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        # Rename P1 columns
        track_cols = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        df = df.rename(columns={c: f"{c}_p1" for c in track_cols})
        df = df.drop(columns=["nfl_player_id"], errors="ignore")

        # Merge Player 2
        df = pd.merge(
            df,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2"),
        )

        # Rename P2 columns explicitly if suffixes didn't catch them (e.g. if name unique)
        for c in track_cols:
            if c in df.columns and not c.endswith("_p1"):
                # This handles cases where column names might collide or not
                # Ideally, suffixes handles it. If column didn't exist in P1, it won't get suffix.
                # But tracking cols are same for both.
                # We force rename just in case.
                pass

        # Clean up P2 columns that might not have been suffixed if P1 didn't have them
        # (Unlikely given identical schema, but good practice)
        curr_cols = df.columns
        for c in track_cols:
            if c in curr_cols:
                df = df.rename(columns={c: f"{c}_p2"})

        df = df.drop(columns=["nfl_player_id"], errors="ignore")

        # Handle Missing Data (e.g., Ground or missing sensor)
        # For Ground, P2 cols are NaN. Fill with 0.
        p1_cols = [f"{c}_p1" for c in track_cols]
        p2_cols = [f"{c}_p2" for c in track_cols]

        df[p1_cols] = df[p1_cols].fillna(0)
        df[p2_cols] = df[p2_cols].fillna(0)

        return df

    def _decompose_vector(self, magnitude, angle_deg):
        """
        Decomposes magnitude and direction (0=Y, 90=X) into vx, vy.
        """
        theta = np.radians(90 - angle_deg)
        vx = magnitude * np.cos(theta)
        vy = magnitude * np.sin(theta)
        return vx, vy

    def _compute_geometric_features(self, df):
        """
        Computes Relative Velocity Aligned features.
        """
        # 1. Decompose Kinematics
        v1_x, v1_y = self._decompose_vector(df["speed_p1"], df["direction_p1"])
        v2_x, v2_y = self._decompose_vector(df["speed_p2"], df["direction_p2"])

        a1_x, a1_y = self._decompose_vector(
            df["acceleration_p1"], df["direction_p1"]
        )  # approx direction
        a2_x, a2_y = self._decompose_vector(df["acceleration_p2"], df["direction_p2"])

        # 2. Relative Vectors
        rx = df["x_position_p1"] - df["x_position_p2"]
        ry = df["y_position_p1"] - df["y_position_p2"]
        vx = v1_x - v2_x
        vy = v1_y - v2_y
        ax = a1_x - a2_x
        ay = a1_y - a2_y

        # 3. Distance & Sentinel Strategy
        dist = np.sqrt(rx**2 + ry**2)
        is_ground = df["nfl_player_id_2"] == "G"
        df["distance"] = np.where(is_ground, self.config.SENTINEL_VALUE, dist)

        # 4. Basis Definition (Relative Velocity)
        v_rel_mag = np.sqrt(vx**2 + vy**2) + 1e-6

        # Unit vector u (Longitudinal)
        ux = vx / v_rel_mag
        uy = vy / v_rel_mag

        # Unit vector u_perp (Transverse) - Rotate 90 deg
        ux_perp = -uy
        uy_perp = ux

        # 5. Projections
        # Longitudinal (Head-on)
        df["r_long"] = rx * ux + ry * uy
        df["a_long"] = ax * ux + ay * uy

        # Transverse (Glancing/Miss)
        df["r_trans"] = rx * ux_perp + ry * uy_perp
        df["a_trans"] = ax * ux_perp + ay * uy_perp

        # 6. Interaction Primitives
        # TTC: Time To Collision. Approx -r_long / v_rel
        df["ttc"] = -df["r_long"] / v_rel_mag

        # Store intermediate vectors for gating if needed, or recompute
        # We'll recompute in gating to keep memory low

        return df

    def _calculate_gating_mask(self, df):
        """
        Implements Relaxed Quadratic Gating.
        Solves for min(d(t)) in window [-1s, 1s].
        """
        # Reconstruct necessary vectors (avoid storing all in DF)
        v1_x, v1_y = self._decompose_vector(df["speed_p1"], df["direction_p1"])
        v2_x, v2_y = self._decompose_vector(df["speed_p2"], df["direction_p2"])
        vx = v1_x - v2_x
        vy = v1_y - v2_y
        v_mag_sq = vx**2 + vy**2

        p1_x, p1_y = df["x_position_p1"], df["y_position_p1"]
        p2_x, p2_y = df["x_position_p2"], df["y_position_p2"]
        rx = p1_x - p2_x
        ry = p1_y - p2_y

        # t_min = -(r . v) / |v|^2
        r_dot_v = rx * vx + ry * vy
        t_min = -r_dot_v / (v_mag_sq + 1e-6)

        # Clamp to window [-1.0, 1.0] seconds (approx 10 steps)
        t_clamped = t_min.clip(-1.0, 1.0)

        # d_min^2 = |r|^2 + 2(r.v)t + |v|^2 t^2
        dist_sq = rx**2 + ry**2
        d_sq_min = dist_sq + 2 * r_dot_v * t_clamped + v_mag_sq * (t_clamped**2)
        d_min = np.sqrt(np.maximum(0, d_sq_min))

        # Mask: Keep if min distance < threshold OR if it's Ground
        mask = (d_min < self.config.GATING_DISTANCE) | (df["nfl_player_id_2"] == "G")

        return mask

    def _create_lagged_features(self, df):
        """
        Generates lagged features for the window [-WINDOW_SIZE, +WINDOW_SIZE].
        """
        # Sort to ensure temporal continuity
        df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

        cols_to_lag = [
            "r_long",
            "r_trans",
            "a_long",
            "a_trans",
            "speed_p1",
            "speed_p2",
            "distance",
            "ttc",
        ]

        # Unique ID for group boundary checking
        pair_ids = (
            df["game_play"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        ).values

        feature_values = df[cols_to_lag].values

        lagged_dfs = []

        # Base DataFrame (Metadata + Target + Gating)
        base_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "gating_active",
        ]
        # Include current features (lag 0)
        lagged_dfs.append(df[base_cols + cols_to_lag])

        shifts = range(-self.config.WINDOW_SIZE, self.config.WINDOW_SIZE + 1)

        for lag in shifts:
            if lag == 0:
                continue

            # Shift arrays
            shifted_vals = np.roll(feature_values, lag, axis=0)
            shifted_ids = np.roll(pair_ids, lag, axis=0)

            # Handle boundaries
            if lag > 0:
                shifted_vals[:lag] = 0
                shifted_ids[:lag] = "INVALID_START"
            else:
                shifted_vals[lag:] = 0
                shifted_ids[lag:] = "INVALID_END"

            # Mask invalid shifts (different pair)
            valid_mask = pair_ids == shifted_ids
            masked_vals = np.where(valid_mask[:, None], shifted_vals, 0)

            # Create DataFrame
            col_names = [f"{c}_lag_{lag}" for c in cols_to_lag]
            lag_df = pd.DataFrame(masked_vals, columns=col_names, index=df.index)
            # Reduce memory immediately
            lag_df = lag_df.astype(np.float32)
            lagged_dfs.append(lag_df)

        # Concatenate all features
        df_final = pd.concat(lagged_dfs, axis=1)

        return df_final
