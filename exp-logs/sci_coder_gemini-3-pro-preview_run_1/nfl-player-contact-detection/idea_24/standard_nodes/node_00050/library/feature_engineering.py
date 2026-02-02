import os
import numpy as np
import pandas as pd
import gc
from library.config import Config
from library.utils import cache_processor, load_metadata, seed_everything


class FeatureEngine:
    """
    Implements the feature engineering pipeline for the VAAM-E strategy.
    Includes Collision-Aligned Vector Decomposition, Spectral Shock features,
    and Relaxed Quadratic Gating.
    """

    def __init__(self):
        seed_everything(Config.SEED)

    def _load_tracking_data(self, path):
        """
        Loads and preprocesses tracking data.
        Converts polar coordinates (speed/direction) to Cartesian vectors.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking data not found at {path}")

        df_track = pd.read_csv(path)

        # Convert Direction (0=North/Y, 90=East/X) to Radians
        # v_x = speed * sin(theta)
        # v_y = speed * cos(theta)
        # Note: NFL tracking data typically has 0 at Y-axis (North) and increases clockwise.

        # Pre-calculate trig values
        rad_dir = np.radians(df_track["direction"].fillna(0))
        rad_orient = np.radians(df_track["orientation"].fillna(0))

        # Velocity Vectors
        df_track["v_x"] = df_track["speed"] * np.sin(rad_dir)
        df_track["v_y"] = df_track["speed"] * np.cos(rad_dir)

        # Acceleration Vectors (using acceleration magnitude and direction)
        # Note: 'acceleration' is magnitude. We assume it aligns with direction of motion
        # or we use finite differences of velocity.
        # Given the dataset description, we'll use the provided 'acceleration' magnitude
        # projected along the motion direction as a primary approximation.
        df_track["a_x"] = df_track["acceleration"] * np.sin(rad_dir)
        df_track["a_y"] = df_track["acceleration"] * np.cos(rad_dir)

        # Orientation Vectors
        df_track["o_x"] = np.sin(rad_orient)
        df_track["o_y"] = np.cos(rad_orient)

        # Select necessary columns to reduce memory usage during merge
        cols_to_keep = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "v_x",
            "v_y",
            "a_x",
            "a_y",
            "o_x",
            "o_y",
            "direction",
            "orientation",
        ]
        return df_track[cols_to_keep]

    def _compute_vector_features(self, df):
        """
        Computes Collision-Aligned Vector Decomposition features.
        Decomposes motion into Radial (impact) and Tangential (shear) components.
        """
        # 1. Relative Position Vector (P1 - P2)
        # We define the collision axis pointing from P2 to P1.
        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Handle division by zero (players at exact same spot)
        dist_safe = dist.replace(0, 1e-6)

        # Unit Vector components (Collision Axis)
        u_x = dx / dist_safe
        u_y = dy / dist_safe

        # 2. Relative Velocity Vector (V1 - V2)
        dv_x = df["v_x_p1"] - df["v_x_p2"]
        dv_y = df["v_y_p1"] - df["v_y_p2"]

        # 3. Relative Acceleration Vector (A1 - A2)
        da_x = df["a_x_p1"] - df["a_x_p2"]
        da_y = df["a_y_p1"] - df["a_y_p2"]

        # 4. Vector Decomposition
        # Radial Component = Dot Product with Unit Vector
        # Tangential Component = Magnitude of (Vector - Radial_Component * Unit_Vector)

        # Radial Velocity (Closing Speed: negative means closing, positive means separating)
        # We invert sign so positive = closing speed (impact intensity)
        # Actually, let's keep standard physics: v . u.
        # If P1 is at (10,0) moving (-1,0) and P2 at (0,0) moving (1,0):
        # dx=10, u=(1,0). dv=(-2,0). dot = -2. Negative = Closing.
        # We will use the raw dot product.
        df["radial_velocity"] = dv_x * u_x + dv_y * u_y

        # Tangential Velocity
        # vt_x = dv_x - (radial_vel * u_x)
        # vt_y = dv_y - (radial_vel * u_y)
        # mag = sqrt(vt_x^2 + vt_y^2)
        # Optimized calculation: |v|^2 = v_rad^2 + v_tan^2 -> v_tan = sqrt(|v|^2 - v_rad^2)
        rel_speed_sq = dv_x**2 + dv_y**2
        df["tangential_velocity"] = np.sqrt(
            np.maximum(0, rel_speed_sq - df["radial_velocity"] ** 2)
        )

        # Radial Acceleration
        df["radial_acceleration"] = da_x * u_x + da_y * u_y

        # Tangential Acceleration
        rel_accel_sq = da_x**2 + da_y**2
        df["tangential_acceleration"] = np.sqrt(
            np.maximum(0, rel_accel_sq - df["radial_acceleration"] ** 2)
        )

        # 5. Time to Collision (TTC)
        # TTC = Distance / Closing Speed.
        # Closing Speed = -Radial Velocity.
        # If separating (Radial Vel > 0), TTC is infinite (set to high value).
        closing_speed = -df["radial_velocity"]
        df["time_to_collision"] = np.where(
            closing_speed > 0.1,  # Avoid div by zero
            dist / closing_speed,
            10.0,  # Cap at 10 seconds
        )

        # 6. Distance
        df["distance"] = dist

        return df

    def _compute_spectral_shock(self, df):
        """
        Computes Spectral Shock features using high-frequency components of acceleration.
        Approximated via Jerk (derivative of acceleration) and Rolling RMS.
        """
        # Sort to ensure temporal order for diff()
        # We need to sort by game_play, pair, and step
        # Create a pair ID for sorting
        df["pair_id"] = (
            df["game_play"]
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        )

        df = df.sort_values(by=["pair_id", "step"])

        # Calculate Jerk (change in radial acceleration)
        # Group by pair_id to prevent bleeding between plays/pairs
        # Using a small window rolling std as a proxy for spectral energy

        # Since groupby is slow on millions of rows, we use a vectorized approach with masking
        # Check if current row belongs to same pair as previous row
        is_same_pair = df["pair_id"] == df["pair_id"].shift(1)

        # Calculate diff
        accel_diff = df["radial_acceleration"].diff().fillna(0)

        # Mask out transitions
        accel_diff = np.where(is_same_pair, accel_diff, 0)

        # Compute "Shock" as the absolute jerk
        df["radial_accel_shock"] = np.abs(accel_diff)

        # Clean up
        df.drop(columns=["pair_id"], inplace=True)

        return df

    def _apply_sentinel_strategy(self, df):
        """
        Applies the Sentinel Value Strategy for Ground interactions.
        Sets distance to Config.SENTINEL_VALUE where nfl_player_id_2 is 'G'.
        """
        mask_ground = df["nfl_player_id_2"] == "G"

        if mask_ground.any():
            # Set Distance Sentinel
            df.loc[mask_ground, "distance"] = Config.SENTINEL_VALUE

            # Zero out vector features for Ground (undefined relative motion)
            # The tree will split on distance=-1 and use scalar P1 features.
            vec_cols = [
                "radial_velocity",
                "tangential_velocity",
                "radial_acceleration",
                "tangential_acceleration",
                "time_to_collision",
                "radial_accel_shock",
            ]
            for col in vec_cols:
                if col in df.columns:
                    df.loc[mask_ground, col] = 0.0

        return df

    def _apply_gating(self, df, is_inference=False):
        """
        Applies Relaxed Quadratic Gating.
        Filters out pairs that are too far away to possibly contact.

        Args:
            df: DataFrame with features.
            is_inference: If True, does NOT drop rows (required for submission format).
                          Instead, it could flag them, but for this task we return all.
        """
        if is_inference:
            return df

        # Gating Logic: Keep if distance < Threshold
        # Note: We must retain Ground interactions (distance = -1.0)
        # Logic: distance <= Threshold OR distance == Sentinel

        mask_keep = (df["distance"] <= Config.GATING_THRESHOLD) | (
            df["distance"] == Config.SENTINEL_VALUE
        )

        # Optional: "Relaxed Quadratic" look-ahead
        # If we implemented the full quadratic solver, we would check min_dist.
        # Here we stick to the robust distance threshold as the primary gate.

        df_gated = df[mask_keep].copy()

        # Print reduction stats
        reduction = 100 * (1 - len(df_gated) / len(df))
        print(f"Gating Reduction: {reduction:.2f}% ({len(df)} -> {len(df_gated)})")

        return df_gated

    @cache_processor
    def process_dataset(
        self, dataset_type, debug=False, load_cached_data=True, cache_path=None
    ):
        """
        Main processing function.
        Loads data, merges tracking, computes features, and applies gating.
        """
        print(f"Processing {dataset_type} dataset (Debug={debug})...")

        # 1. Determine Paths
        if dataset_type == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            track_path = Config.TRAIN_TRACKING_PATH
        elif dataset_type == "val":
            meta_path = Config.VAL_METADATA_PATH
            track_path = Config.TRAIN_TRACKING_PATH  # Val uses train tracking
        elif dataset_type == "test":
            meta_path = Config.TEST_METADATA_PATH
            track_path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        # 2. Load Metadata
        df_meta = load_metadata(
            meta_path, debug=debug, sample_size=Config.DEBUG_SAMPLE_SIZE
        )

        # 3. Load Tracking Data
        df_track = self._load_tracking_data(track_path)

        # 4. Merge Player 1 Tracking
        # Ensure types match
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_track["game_play"] = df_track["game_play"].astype(str)

        # Merge P1
        df_merged = pd.merge(
            df_meta,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1"),
        )
        # Rename columns that didn't get a suffix (from the left join collision)
        # The merge operation puts suffix on overlapping columns.
        # Unique columns from track need renaming manually if they didn't collide
        # But here we want explicit _p1 suffixes.
        # Let's rename tracking df columns before merge to be safe?
        # No, standard merge is fine, but we need to ensure we know which is which.
        # The merge above adds _p1 to tracking cols if they exist in meta (which they don't mostly).
        # So we rename manually.

        track_cols = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_map_p1 = {c: f"{c}_p1" for c in track_cols}
        df_merged.rename(columns=rename_map_p1, inplace=True)

        # 5. Merge Player 2 Tracking
        # Separate Ground ('G') from Players
        mask_ground = df_merged["nfl_player_id_2"] == "G"

        # We can't merge 'G' with tracking. So we merge on the subset that is NOT 'G'.
        # However, splitting and recombining is messy.
        # Strategy: Create a temporary merge key. If 'G', set to -999.
        df_merged["merge_id_2"] = df_merged["nfl_player_id_2"]
        df_merged.loc[mask_ground, "merge_id_2"] = -999
        df_merged["merge_id_2"] = df_merged["merge_id_2"].astype(
            int
        )  # Now safe to cast

        # Merge P2
        df_merged = pd.merge(
            df_merged,
            df_track,
            left_on=["game_play", "step", "merge_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2"),
        )

        # Rename P2 columns
        # Note: If columns already had _p1, they won't conflict.
        # But original tracking cols (if any leaked) might.
        # The tracking cols from the second merge will have no suffix or _p2 depending on collision with existing.
        # Since df_merged already has _p1 columns, the raw tracking cols (x_position etc) are not there.
        # So the new columns come in as 'x_position', 'y_position' etc.
        rename_map_p2 = {c: f"{c}_p2" for c in track_cols}
        df_merged.rename(columns=rename_map_p2, inplace=True)

        # Fill NaNs for P2 (Ground or missing tracking) with 0
        p2_cols = list(rename_map_p2.values())
        df_merged[p2_cols] = df_merged[p2_cols].fillna(0.0)

        # Fill NaNs for P1 (missing tracking) with 0
        p1_cols = list(rename_map_p1.values())
        df_merged[p1_cols] = df_merged[p1_cols].fillna(0.0)

        # 6. Feature Engineering
        print("Computing Vector Features...")
        df_features = self._compute_vector_features(df_merged)

        print("Computing Spectral Shock...")
        df_features = self._compute_spectral_shock(df_features)

        print("Applying Sentinel Strategy...")
        df_features = self._apply_sentinel_strategy(df_features)

        # 7. Gating
        # Only gate if not test set (inference)
        is_inference = dataset_type == "test"
        df_final = self._apply_gating(df_features, is_inference=is_inference)

        # 8. Select Final Columns
        # Keep metadata + features
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        # Ensure contact exists (test set has it as 0)
        if "contact" not in df_final.columns:
            df_final["contact"] = 0

        final_cols = meta_cols + Config.FEATURES

        # Deduplicate columns (e.g., 'step' appears in both meta and features)
        final_cols = list(dict.fromkeys(final_cols))

        # Filter columns that exist
        final_cols = [c for c in final_cols if c in df_final.columns]

        return df_final[final_cols]
