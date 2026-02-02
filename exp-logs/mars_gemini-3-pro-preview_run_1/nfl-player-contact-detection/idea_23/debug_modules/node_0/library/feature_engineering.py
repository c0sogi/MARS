import os
import hashlib
import numpy as np
import pandas as pd
from library.config import Config


class FeatureEngineer:
    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, metadata_path, tracking_path, mode):
        """Generates a unique cache file path based on inputs and configuration."""
        # Include gating distance in hash to invalidate cache if parameter changes
        hash_str = (
            f"{metadata_path}_{tracking_path}_{mode}_{self.config.GATING_DISTANCE}"
        )
        hash_hex = hashlib.md5(hash_str.encode()).hexdigest()
        filename = f"features_{mode}_{hash_hex}.parquet"
        return os.path.join(self.cache_dir, filename)

    def load_tracking(self, tracking_path):
        """Loads tracking data with optimized types."""
        # Specifying types can save memory, but pandas inference is usually fine for this size
        df = pd.read_csv(tracking_path)
        return df

    def merge_data(self, df_meta, df_track):
        """
        Merges metadata with tracking data for both players.
        Separates Ground interactions.
        """
        # Ensure consistent types for merging
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

        df_track["game_play"] = df_track["game_play"].astype(str)
        df_track["step"] = df_track["step"].astype(int)
        df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(int)

        # Split Player-Player and Player-Ground
        # Ground is indicated by 'G' in nfl_player_id_2
        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_ground = df_meta[mask_ground].copy()
        df_players = df_meta[~mask_ground].copy()

        # Process Player-Player interactions
        if not df_players.empty:
            df_players["nfl_player_id_2"] = df_players["nfl_player_id_2"].astype(int)

            # Merge Player 1 Tracking
            df_players = df_players.merge(
                df_track,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p1"),
            )

            # Explicit rename to ensure p1 columns exist even if no collision occurred
            # (Pandas only adds suffix if collision, so we force rename for consistency)
            cols_to_rename = {
                "x_position": "x_position_p1",
                "y_position": "y_position_p1",
                "speed": "speed_p1",
                "acceleration": "acceleration_p1",
                "direction": "direction_p1",
                "orientation": "orientation_p1",
                "sa": "sa_p1",
            }
            # Only rename columns that exist and haven't been suffixed yet
            curr_cols = df_players.columns
            rename_map = {k: v for k, v in cols_to_rename.items() if k in curr_cols}
            df_players = df_players.rename(columns=rename_map)
            df_players = df_players.drop(columns=["nfl_player_id"], errors="ignore")

            # Merge Player 2 Tracking
            df_players = df_players.merge(
                df_track,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p2"),
            )

            cols_to_rename_2 = {
                "x_position": "x_position_p2",
                "y_position": "y_position_p2",
                "speed": "speed_p2",
                "acceleration": "acceleration_p2",
                "direction": "direction_p2",
                "orientation": "orientation_p2",
                "sa": "sa_p2",
            }
            curr_cols = df_players.columns
            rename_map_2 = {k: v for k, v in cols_to_rename_2.items() if k in curr_cols}
            df_players = df_players.rename(columns=rename_map_2)
            df_players = df_players.drop(columns=["nfl_player_id"], errors="ignore")

        return df_players, df_ground

    def decompose_vectors(self, df):
        """
        Performs Vector Decomposition to extract Radial and Tangential kinematics.
        """
        # 1. Relative Position Vector
        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        distance = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero for unit vectors
        dist_safe = distance.replace(0, 1e-6)

        # Unit Collision Vector (pointing P2 -> P1)
        rx = dx / dist_safe
        ry = dy / dist_safe

        # 2. Velocity Decomposition
        # Convert Speed/Direction to Velocity Components
        # NFL Data: 0 deg is Y-axis (North), 90 deg is X-axis (East)?
        # Standard conversion assuming direction is in degrees:
        # We use standard trig; exact orientation convention cancels out in relative calc
        # if consistent for both players.
        def get_velocity_components(speed, direction):
            rad = np.radians(direction.fillna(0))
            # Assuming standard math convention for simplicity of relative calc
            # (Rotation doesn't affect magnitude of relative components)
            vx = speed * np.sin(rad)
            vy = speed * np.cos(rad)
            return vx, vy

        vx1, vy1 = get_velocity_components(df["speed_p1"], df["direction_p1"])
        vx2, vy2 = get_velocity_components(df["speed_p2"], df["direction_p2"])

        # Relative Velocity
        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Radial Velocity (Impact Speed): Dot product of v_rel and r_hat
        v_rad = dvx * rx + dvy * ry

        # Tangential Velocity (Shear Speed): Magnitude of rejection
        # v_tan^2 = |v_rel|^2 - v_rad^2
        v_rel_sq = dvx**2 + dvy**2
        v_tan = np.sqrt(np.maximum(0, v_rel_sq - v_rad**2))

        # 3. Acceleration Decomposition
        # We approximate acceleration vector direction using velocity direction
        # (assuming acceleration is primarily tangential to path, i.e., speeding up/slowing down)
        # This is a simplification as we lack explicit lateral acceleration data.
        s1_safe = df["speed_p1"].replace(0, 1.0)
        s2_safe = df["speed_p2"].replace(0, 1.0)

        # Project scalar acceleration onto velocity vector to get accel vector approx
        ax1 = df["acceleration_p1"] * (vx1 / s1_safe)
        ay1 = df["acceleration_p1"] * (vy1 / s1_safe)
        ax2 = df["acceleration_p2"] * (vx2 / s2_safe)
        ay2 = df["acceleration_p2"] * (vy2 / s2_safe)

        dax = ax1 - ax2
        day = ay1 - ay2

        # Radial Acceleration
        a_rad = dax * rx + day * ry

        # Tangential Acceleration
        a_rel_sq = dax**2 + day**2
        a_tan = np.sqrt(np.maximum(0, a_rel_sq - a_rad**2))

        # 4. Derived Physics Primitives
        # Time to Collision: distance / closing_speed
        # closing_speed = -v_rad
        closing_speed = -v_rad
        ttc = distance / closing_speed.replace(0, 0.001)
        # Clip TTC for stability: if moving away (closing_speed < 0), set to high value
        ttc = np.where(closing_speed <= 0, 10.0, ttc)
        ttc = np.clip(ttc, 0, 10.0)

        # Transient Spectral Energy Proxy
        # Using squared radial acceleration as instantaneous energy proxy
        radial_energy = a_rad**2

        # Assign to DataFrame
        df["distance"] = distance
        df["radial_velocity"] = v_rad
        df["tangential_velocity"] = v_tan
        df["radial_acceleration"] = a_rad
        df["tangential_acceleration"] = a_tan
        df["time_to_collision"] = ttc
        df["radial_accel_energy"] = radial_energy

        return df

    def apply_quadratic_gating(self, df):
        """
        Applies Relaxed Quadratic Gating.
        Projects distance d(t) = d + v*t + 0.5*a*t^2 over a window.
        Keeps rows where min(d(t)) < GATING_DISTANCE.
        """
        d = df["distance"]
        v = df["radial_velocity"]
        a = df["radial_acceleration"]

        # Check specific time points in the future/past window [-0.5, 0.5]
        # This accounts for synchronization errors and near-future contact
        time_points = [0.0, -0.2, -0.5, 0.2, 0.5]

        min_projected_dist = d.copy()

        for t in time_points:
            d_t = d + v * t + 0.5 * a * (t**2)
            min_projected_dist = np.minimum(min_projected_dist, d_t)

        # Filter
        mask = min_projected_dist < self.config.GATING_DISTANCE
        return df[mask].copy()

    def generate_features(
        self, metadata_path, tracking_path, mode="train", load_cached_data=True
    ):
        """
        Main pipeline execution function.
        """
        # 1. Check Cache
        cache_path = self._get_cache_path(metadata_path, tracking_path, mode)
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating features for {mode} from scratch...")

        # 2. Load Data
        df_meta = pd.read_csv(metadata_path)
        df_track = self.load_tracking(tracking_path)

        # 3. Merge
        df_players, df_ground = self.merge_data(df_meta, df_track)

        # 4. Process Players (Physics + Gating)
        if not df_players.empty:
            df_players = self.decompose_vectors(df_players)

            # Apply Gating
            # We apply gating to all sets. For test set, this acts as a filter.
            # Downstream prediction logic must handle missing rows (fill 0).
            original_count = len(df_players)
            df_players = self.apply_quadratic_gating(df_players)
            print(
                f"Gating reduced player pairs from {original_count} to {len(df_players)}"
            )

        # 5. Process Ground (Sentinel Assignment)
        # Ground is always kept (distance -1 < 3.0)
        sentinel = self.config.SENTINEL_VALUE
        for feature in self.config.FEATURES:
            if feature not in df_ground.columns:
                df_ground[feature] = sentinel

        # Ensure 'distance' is sentinel
        df_ground["distance"] = sentinel

        # 6. Combine
        df_final = pd.concat([df_players, df_ground], axis=0, ignore_index=True)

        # Fill any remaining NaNs (e.g., from missing tracking data) with 0
        df_final = df_final.fillna(0)

        # 7. Save Cache
        print(f"Saving features to {cache_path}")
        df_final.to_parquet(cache_path)

        return df_final
