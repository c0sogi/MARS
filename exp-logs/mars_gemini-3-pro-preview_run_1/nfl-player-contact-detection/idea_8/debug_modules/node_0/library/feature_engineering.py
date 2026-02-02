import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    setup_logger,
    reduce_mem_usage,
    save_to_parquet,
    load_from_parquet,
    get_experiment_hash,
)


class FeatureGenerator:
    """
    Implements the Variable-Resolution Feature Engineering pipeline.
    Handles the generation of Kinematic, Physics, and Contextual features
    with support for Multi-Fidelity Tiers (Scout vs. Expert).
    """

    def __init__(self):
        self.logger = setup_logger(name="FeatureGenerator")
        self.window_size = Config.WINDOW_SIZE

    def _compute_angular_diff(self, a, b):
        """
        Computes the shortest angular difference between two angles in degrees.
        Result is in range [-180, 180].
        """
        return (a - b + 180) % 360 - 180

    def _add_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes base kinematic features for player pairs.
        Handles Player-Ground interactions where P2 features are NaN.
        """
        # Distance
        # If P2 is Ground (NaN coordinates), distance remains NaN (or handled by model)
        df["distance"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        )

        # Speed Difference
        df["speed_diff"] = np.abs(df["speed_p1"] - df["speed_p2"])

        # Acceleration Difference
        df["acc_p1"] = df["acceleration_p1"]  # Alias for consistency
        df["acc_p2"] = df["acceleration_p2"]
        df["acc_diff"] = np.abs(df["acc_p1"] - df["acc_p2"])

        # Orientation & Direction
        df["orient_p1"] = df["orientation_p1"]
        df["orient_p2"] = df["orientation_p2"]
        df["dir_p1"] = df["direction_p1"]
        df["dir_p2"] = df["direction_p2"]

        # Angular Differences
        df["orient_diff"] = np.abs(
            self._compute_angular_diff(df["orientation_p1"], df["orientation_p2"])
        )
        df["dir_diff"] = np.abs(
            self._compute_angular_diff(df["direction_p1"], df["direction_p2"])
        )

        return df

    def _add_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes physics derivatives (Jerk) using temporal shifts.
        Assumes df is sorted by game_play and step.
        """
        # We need to ensure we don't shift across different plays.
        # Create a mask for valid transitions (same play, consecutive step)
        # Note: step is int, increments by 1.

        # Shift 1 step back to get t-1
        prev_game_play = df["game_play"].shift(1)
        prev_step = df["step"].shift(1)

        # Valid mask: same play, step is current - 1
        valid_mask = (df["game_play"] == prev_game_play) & (df["step"] == prev_step + 1)

        # Helper to compute derivative
        def compute_derivative(col_name):
            prev_val = df[col_name].shift(1)
            # Derivative = (Current - Prev) / 0.1s.
            # We just store the raw diff as the scale is constant.
            diff = df[col_name] - prev_val
            return diff.where(valid_mask, np.nan)

        # Jerk (Derivative of Acceleration)
        df["jerk_p1"] = compute_derivative("acc_p1")
        df["jerk_p2"] = compute_derivative("acc_p2")

        # Angular Jerk (Derivative of Orientation ~ Angular Velocity change?
        # Actually Orientation derivative is Angular Velocity. Derivative of that is Ang Jerk.
        # Given we have Orientation, let's compute Angular Velocity first then Jerk.
        # Or just 'Angular Change' as a proxy for impulse.
        # We will compute simple diff of orientation for now as 'angular_jerk' per config names.

        # Handle cyclic diff for angles
        prev_orient_p1 = df["orient_p1"].shift(1)
        prev_orient_p2 = df["orient_p2"].shift(1)

        diff_p1 = self._compute_angular_diff(df["orient_p1"], prev_orient_p1)
        diff_p2 = self._compute_angular_diff(df["orient_p2"], prev_orient_p2)

        df["angular_jerk_p1"] = diff_p1.where(valid_mask, np.nan)
        df["angular_jerk_p2"] = diff_p2.where(valid_mask, np.nan)

        return df

    def _add_context(self, df: pd.DataFrame, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes spatial context (density and cluster speed) by joining with raw tracking data.
        """
        self.logger.info("Computing spatial context features...")

        # 1. Prepare Tracking Data for Join
        # We need (game_play, step, x, y, speed) for all players
        track_cols = ["game_play", "step", "x_position", "y_position", "speed"]
        track_sub = tracking_df[track_cols].copy()
        track_sub.rename(
            columns={
                "x_position": "x_other",
                "y_position": "y_other",
                "speed": "speed_other",
            },
            inplace=True,
        )

        # 2. Prepare Base DF for Join
        # We need to map context back to the original rows. Use index.
        df_idx = df.reset_index()
        base_cols = [
            "index",
            "game_play",
            "step",
            "x_position_p1",
            "y_position_p1",
            "x_position_p2",
            "y_position_p2",
        ]
        base_sub = df_idx[base_cols]

        # 3. Merge to get all players for each interaction frame
        # This expands the dataframe: 1 row per interaction -> N rows (N=players on field)
        merged = pd.merge(base_sub, track_sub, on=["game_play", "step"], how="inner")

        # 4. Compute Context for Player 1
        dx_p1 = merged["x_position_p1"] - merged["x_other"]
        dy_p1 = merged["y_position_p1"] - merged["y_other"]
        dist_p1 = np.sqrt(dx_p1**2 + dy_p1**2)

        # Filter neighbors: within 2 yards, excluding self (dist > 0.01 to avoid float errors)
        is_neighbor_p1 = (dist_p1 < 2.0) & (dist_p1 > 0.01)

        # 5. Compute Context for Player 2
        # If P2 is Ground (NaN), dist will be NaN, condition False. Correct.
        dx_p2 = merged["x_position_p2"] - merged["x_other"]
        dy_p2 = merged["y_position_p2"] - merged["y_other"]
        dist_p2 = np.sqrt(dx_p2**2 + dy_p2**2)

        is_neighbor_p2 = (dist_p2 < 2.0) & (dist_p2 > 0.01)

        # 6. Aggregate
        # We calculate density (count) and cluster speed (mean speed of neighbors)

        # Mask non-neighbors for speed calculation
        speed_p1_neighbors = merged["speed_other"].where(is_neighbor_p1, np.nan)
        speed_p2_neighbors = merged["speed_other"].where(is_neighbor_p2, np.nan)

        # Group by original index
        # We use the 'index' column from reset_index() to map back correctly
        grouped = merged.groupby("index")

        context_features = pd.DataFrame(
            {
                "spatial_density_p1": grouped.apply(
                    lambda x: is_neighbor_p1.loc[x.index].sum()
                ),
                "cluster_speed_p1": grouped["speed_other"]
                .apply(lambda x: x[is_neighbor_p1.loc[x.index]].mean())
                .fillna(0),
                "spatial_density_p2": grouped.apply(
                    lambda x: is_neighbor_p2.loc[x.index].sum()
                ),
                "cluster_speed_p2": grouped["speed_other"]
                .apply(lambda x: x[is_neighbor_p2.loc[x.index]].mean())
                .fillna(0),
            }
        )

        # 7. Merge back to original DF
        # Combine P1 and P2 context into single features representing the "Interaction Context"
        # We sum densities and average speeds for the pair context
        df = df.join(context_features)

        df["spatial_density"] = df["spatial_density_p1"] + df["spatial_density_p2"]
        # Average cluster speed, handling zeros
        df["cluster_speed"] = (df["cluster_speed_p1"] + df["cluster_speed_p2"]) / 2.0

        # Cleanup intermediate columns
        df.drop(
            columns=[
                "spatial_density_p1",
                "spatial_density_p2",
                "cluster_speed_p1",
                "cluster_speed_p2",
            ],
            inplace=True,
        )

        del merged, track_sub, base_sub, context_features
        gc.collect()

        return df

    def _create_windows(
        self, df: pd.DataFrame, features_to_window: list
    ) -> pd.DataFrame:
        """
        Generates lag and lead features for the specified columns.
        """
        self.logger.info(
            f"Creating temporal windows (+/- {self.window_size}) for {len(features_to_window)} features..."
        )

        # Ensure sorting
        df = df.sort_values(["game_play", "step"])

        # Pre-calculate play transition masks
        # We need to know if row[i] and row[i+lag] belong to the same play
        game_play_arr = df["game_play"].values

        for col in features_to_window:
            if col not in df.columns:
                continue

            col_values = df[col].values

            for lag in range(1, self.window_size + 1):
                # --- Lag (Past) ---
                shifted_vals = df[col].shift(lag)
                # Check play boundary
                # We can check if game_play == game_play.shift(lag)
                mask = game_play_arr == np.roll(game_play_arr, lag)
                # Fix roll wrap-around at the beginning
                mask[:lag] = False

                df[f"{col}_lag{lag}"] = shifted_vals.where(mask, np.nan)

                # --- Lead (Future) ---
                shifted_vals_lead = df[col].shift(-lag)
                mask_lead = game_play_arr == np.roll(game_play_arr, -lag)
                # Fix roll wrap-around at the end
                mask_lead[-lag:] = False

                df[f"{col}_lead{lag}"] = shifted_vals_lead.where(mask_lead, np.nan)

        return df

    def generate(
        self,
        merged_df: pd.DataFrame,
        tracking_df: pd.DataFrame,
        tier: int = 1,
        split: str = "train",
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Main entry point for feature generation.

        Args:
            merged_df: The merged metadata + tracking (P1/P2) dataframe.
            tracking_df: The raw tracking dataframe (needed for context).
            tier: 1 (Scout) or 2 (Expert).
            split: 'train', 'val', or 'test'.
            load_cached_data: Whether to use caching.

        Returns:
            pd.DataFrame: Dataframe with generated features.
        """
        # 1. Generate Cache Key
        # We include the shape of input to invalidate cache if data changes
        params = {
            "tier": tier,
            "split": split,
            "window_size": self.window_size,
            "input_shape": merged_df.shape,
            "tracking_shape": tracking_df.shape if tracking_df is not None else 0,
        }
        cache_hash = get_experiment_hash(params)
        cache_filename = f"features_{split}_tier{tier}_{cache_hash}.parquet"

        # 2. Check Cache
        if load_cached_data:
            cached_df = load_from_parquet(cache_filename)
            if cached_df is not None:
                self.logger.info(f"Loaded Tier {tier} features for {split} from cache.")
                return cached_df

        self.logger.info(f"Generating Tier {tier} features for {split} from scratch...")

        # 3. Preprocessing
        # Sort is critical for windowing and physics
        df = merged_df.sort_values(["game_play", "step"]).reset_index(drop=True)

        # 4. Base Feature Computation
        df = self._add_kinematics(df)
        df = self._add_physics(df)

        if tracking_df is not None:
            df = self._add_context(df, tracking_df)
        else:
            self.logger.warning(
                "Tracking data not provided. Skipping Context features."
            )
            df["spatial_density"] = 0
            df["cluster_speed"] = 0

        # 5. Windowing Strategy (Variable Resolution)
        # Determine which features to window based on Tier

        # Always window Kinematics & Physics
        feats_to_window = Config.KINEMATIC_FEATURES_BASE + Config.PHYSICS_FEATURES_BASE

        # Only window Context for Tier 2 (Expert)
        if tier == 2:
            feats_to_window += Config.CONTEXT_FEATURES_BASE

        df = self._create_windows(df, feats_to_window)

        # 6. Feature Selection
        # Filter to only the columns defined in Config for this Tier
        # We also keep identifiers and target for the training loop
        required_features = Config.get_feature_list(tier)
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]

        # Handle test set which might not have 'contact'
        available_cols = df.columns.tolist()
        final_cols = [c for c in meta_cols if c in available_cols] + [
            c for c in required_features if c in available_cols
        ]

        df_final = df[final_cols].copy()

        # 7. Optimization & Saving
        df_final = reduce_mem_usage(df_final)
        save_to_parquet(df_final, cache_filename)

        self.logger.info(
            f"Tier {tier} feature generation complete. Shape: {df_final.shape}"
        )
        return df_final
