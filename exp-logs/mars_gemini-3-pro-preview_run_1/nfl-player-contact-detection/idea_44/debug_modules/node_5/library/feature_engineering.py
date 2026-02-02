import os
import numpy as np
import pandas as pd
import gc
from library.config import Config
from library.utils import setup_logging, CacheManager


class KinematicFeatureEngine:
    """
    Implements the Kinematically-Aligned Relative-Physics feature engineering pipeline (KARP-AM).
    Handles data merging, windowing, basis projection, and interaction primitive calculation.
    """

    def __init__(self):
        self.logger = setup_logging()
        self.cache = CacheManager()
        self.window_size = Config.WINDOW_SIZE
        self.gating_dist = Config.GATING_DIST
        self.ground_sentinel = Config.GROUND_DIST_SENTINEL

    def process_data(
        self,
        metadata_df: pd.DataFrame,
        tracking_df: pd.DataFrame,
        dataset_key: str,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Main entry point for feature processing.
        Checks cache, processes data if needed, and returns the feature set.
        """
        # Define cache key based on data shape and configuration
        params = {
            "dataset_key": dataset_key,
            "window_size": self.window_size,
            "gating_dist": self.gating_dist,
            "meta_shape": metadata_df.shape,
            "track_shape": tracking_df.shape,
            "strategy": "KARP_AM_v1",
        }
        cache_file = (
            self.cache.generate_key(f"features_{dataset_key}", params) + ".parquet"
        )

        if load_cached_data and self.cache.exists(cache_file):
            self.logger.info(f"Loading cached features from {cache_file}")
            return self.cache.load(cache_file)

        self.logger.info(
            f"Generating features for {dataset_key} (Cache miss or force reload)..."
        )

        # 1. Preprocess Tracking Data (Calculate Vx, Vy, Lags)
        track_wide = self._prepare_tracking_data(tracking_df)

        # 2. Merge and Compute Features
        features = self._compute_features(metadata_df, track_wide)

        # 3. Save to cache
        self.logger.info(f"Saving features to {cache_file}")
        self.cache.save(features, cache_file)

        return features

    def _prepare_tracking_data(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares tracking data: converts polar vectors to cartesian, computes jerk, generates temporal lags.
        Returns a wide DataFrame where each row contains the window of data for a (play, player, step).
        """
        self.logger.info("Preprocessing tracking data...")
        df = tracking_df.copy()

        # Convert Speed/Dir to Vx, Vy
        # Assumption: dir is in degrees. Standard NFL orientation: 0 is Y-axis (North), 90 is X-axis (East).
        # vx = speed * sin(rad), vy = speed * cos(rad)
        rad = np.radians(df["direction"].fillna(0))
        df["vx"] = df["speed"] * np.sin(rad)
        df["vy"] = df["speed"] * np.cos(rad)

        # Compute Cartesian Acceleration components
        # We assume acceleration magnitude aligns with motion direction for simplicity in this dataset
        df["ax"] = df["acceleration"] * np.sin(rad)
        df["ay"] = df["acceleration"] * np.cos(rad)

        # Compute Jerk (Derivative of Acceleration Magnitude)
        # Sort first to ensure correct diff
        df = df.sort_values(["game_play", "nfl_player_id", "step"])
        df["jerk"] = (
            df.groupby(["game_play", "nfl_player_id"])["acceleration"].diff().fillna(0)
        )

        # Select columns to lag
        cols_to_lag = [
            "x_position",
            "y_position",
            "vx",
            "vy",
            "ax",
            "ay",
            "jerk",
            "orientation",
        ]

        # Generate Lags
        # We need t-10 to t+10.
        # Using shift(-k) brings data from t+k to the current row t.
        grouper = df.groupby(["game_play", "nfl_player_id"])

        result_df = df[["game_play", "nfl_player_id", "step"]].copy()

        # Create wide format
        for k in range(-self.window_size, self.window_size + 1):
            suffix = f"_{k}"
            shifted = grouper[cols_to_lag].shift(-k)
            shifted.columns = [c + suffix for c in cols_to_lag]
            result_df = pd.concat([result_df, shifted], axis=1)

        return result_df

    def _compute_features(
        self, meta_df: pd.DataFrame, track_wide: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges metadata with wide tracking data and computes kinematic features.
        """
        self.logger.info("Merging and computing kinematic features...")

        # Split Meta into Player-Player and Player-Ground
        meta_pg = meta_df[meta_df["nfl_player_id_2"] == "G"].copy()
        meta_pp = meta_df[meta_df["nfl_player_id_2"] != "G"].copy()

        features_list = []

        # --- Process Player-Player ---
        if not meta_pp.empty:
            meta_pp["nfl_player_id_2"] = meta_pp["nfl_player_id_2"].astype(int)

            # Merge P1
            df_pp = meta_pp.merge(
                track_wide,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="inner",
            )
            # Rename P1 cols
            rename_p1 = {
                c: f"{c}_p1"
                for c in track_wide.columns
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_pp = df_pp.rename(columns=rename_p1).drop(columns=["nfl_player_id"])

            # Merge P2
            df_pp = df_pp.merge(
                track_wide,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="inner",
            )
            # Rename P2 cols
            rename_p2 = {
                c: f"{c}_p2"
                for c in track_wide.columns
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_pp = df_pp.rename(columns=rename_p2).drop(columns=["nfl_player_id"])

            # Compute PP Features
            feat_pp = self._calculate_kinematics(df_pp, mode="PP")
            features_list.append(feat_pp)

        # --- Process Player-Ground ---
        if not meta_pg.empty:
            # Merge P1 only
            df_pg = meta_pg.merge(
                track_wide,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="inner",
            )
            rename_p1 = {
                c: f"{c}_p1"
                for c in track_wide.columns
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_pg = df_pg.rename(columns=rename_p1).drop(columns=["nfl_player_id"])

            # Compute PG Features
            feat_pg = self._calculate_kinematics(df_pg, mode="PG")
            features_list.append(feat_pg)

        # Combine
        if not features_list:
            self.logger.warning("No features generated. Check data consistency.")
            return pd.DataFrame()

        full_features = pd.concat(features_list, axis=0).reset_index(drop=True)

        # Fill NaNs (some lags might be missing at start/end of plays)
        full_features = full_features.fillna(0)

        return full_features

    def _calculate_kinematics(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Vectorized physics engine.
        mode: 'PP' (Player-Player) or 'PG' (Player-Ground)
        """
        # Output container
        base_cols = ["contact_id", "game_play", "step"]
        if "contact" in df.columns:
            base_cols.append("contact")
        out = df[base_cols].copy()

        # Container for gating logic
        min_dists = []

        for k in range(-self.window_size, self.window_size + 1):
            suffix = f"_{k}"

            # P1 Data
            x1 = df[f"x_position{suffix}_p1"]
            y1 = df[f"y_position{suffix}_p1"]
            vx1 = df[f"vx{suffix}_p1"]
            vy1 = df[f"vy{suffix}_p1"]
            ax1 = df[f"ax{suffix}_p1"]
            ay1 = df[f"ay{suffix}_p1"]

            if mode == "PP":
                # P2 Data
                x2 = df[f"x_position{suffix}_p2"]
                y2 = df[f"y_position{suffix}_p2"]
                vx2 = df[f"vx{suffix}_p2"]
                vy2 = df[f"vy{suffix}_p2"]
                ax2 = df[f"ax{suffix}_p2"]
                ay2 = df[f"ay{suffix}_p2"]

                # Relative State
                rx = x1 - x2
                ry = y1 - y2
                rvx = vx1 - vx2
                rvy = vy1 - vy2
                rax = ax1 - ax2
                ray = ay1 - ay2

                dist = np.sqrt(rx**2 + ry**2)
                min_dists.append(dist)

            else:  # PG
                # Ground Sentinel Strategy
                dist = pd.Series(np.full(len(df), self.ground_sentinel), index=df.index)
                min_dists.append(dist)

                # Relative Velocity is just P1 velocity (relative to static ground)
                # Relative Position is undefined for ground, set to 0
                rx = pd.Series(np.zeros(len(df)), index=df.index)
                ry = pd.Series(np.zeros(len(df)), index=df.index)
                rvx = vx1
                rvy = vy1
                rax = ax1
                ray = ay1

            # --- Basis Projection ---
            # Basis u = v_rel / |v_rel|
            v_rel_mag = np.sqrt(rvx**2 + rvy**2) + 1e-6  # Avoid div/0
            ux = rvx / v_rel_mag
            uy = rvy / v_rel_mag

            # Orthogonal Basis u_perp (-uy, ux)
            ux_perp = -uy
            uy_perp = ux

            # Projections
            # Position (only for PP, Sentinel for PG)
            if mode == "PP":
                r_long = rx * ux + ry * uy
                r_trans = rx * ux_perp + ry * uy_perp
            else:
                r_long = pd.Series(
                    np.full(len(df), self.ground_sentinel), index=df.index
                )
                r_trans = pd.Series(
                    np.full(len(df), self.ground_sentinel), index=df.index
                )

            # Acceleration
            a_long = rax * ux + ray * uy
            a_trans = rax * ux_perp + ray * uy_perp

            # --- Interaction Primitives ---
            # TTC = dist / closing_speed
            # closing_speed = - (v_rel . r_hat)
            if mode == "PP":
                dot = rx * rvx + ry * rvy
                # closing speed is positive if moving closer
                closing_speed = -(dot) / (dist + 1e-6)
                ttc = dist / (closing_speed + 1e-6)
                ttc = ttc.clip(-5, 5)  # Clip to reasonable physics range
            else:
                ttc = pd.Series(np.zeros(len(df)), index=df.index)

            # Store Features
            out[f"dist{suffix}"] = dist
            out[f"r_long{suffix}"] = r_long
            out[f"r_trans{suffix}"] = r_trans
            out[f"v_rel{suffix}"] = v_rel_mag
            out[f"a_long{suffix}"] = a_long
            out[f"a_trans{suffix}"] = a_trans
            out[f"ttc{suffix}"] = ttc

            # Raw P1/P2 magnitudes (useful context)
            out[f"s1{suffix}"] = np.sqrt(vx1**2 + vy1**2)
            out[f"a1{suffix}"] = np.sqrt(ax1**2 + ay1**2)
            out[f"jerk1{suffix}"] = df[f"jerk{suffix}_p1"]

            if mode == "PP":
                out[f"s2{suffix}"] = np.sqrt(vx2**2 + vy2**2)
                out[f"a2{suffix}"] = np.sqrt(ax2**2 + ay2**2)
                out[f"jerk2{suffix}"] = df[f"jerk{suffix}_p2"]
            else:
                out[f"s2{suffix}"] = 0
                out[f"a2{suffix}"] = 0
                out[f"jerk2{suffix}"] = 0

        # --- Gating (Stage 0) ---
        if mode == "PP":
            # Calculate min distance across the window
            all_dists = pd.concat(min_dists, axis=1)
            min_dist_val = all_dists.min(axis=1)

            # Filter: Keep pairs where min(d(t)) < GATING_DIST
            mask = min_dist_val < self.gating_dist
            dropped_count = len(out) - mask.sum()
            if dropped_count > 0:
                self.logger.info(
                    f"Gating (PP): Dropping {dropped_count} / {len(out)} pairs with min_dist >= {self.gating_dist}"
                )
            out = out[mask].copy()

        return out
