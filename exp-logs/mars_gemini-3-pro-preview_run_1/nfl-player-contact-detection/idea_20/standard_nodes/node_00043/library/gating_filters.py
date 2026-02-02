import numpy as np
import pandas as pd
import os
from library.config import GATING_THRESHOLD, SENTINEL_VALUE, IDEA_DIR
from library.utils import CacheManager, setup_logger


class GatingFilter:
    """
    Implements the Relaxed Quadratic Reachability Gating.
    Filters player pairs based on whether they can theoretically reach a distance
    less than GATING_THRESHOLD within a 1.0s window, using a quadratic motion model.
    """

    def __init__(self):
        self.logger = setup_logger("gating_filter")
        self.cache_manager = CacheManager(cache_dir=IDEA_DIR)

    def _calculate_projected_kinematics(self, df):
        """
        Calculates the initial distance (d0), relative velocity (v_rel),
        and relative acceleration (a_rel) projected onto the connecting vector.
        """
        # Convert degrees to radians
        # Note: We assume standard trigonometric mapping. Since we only care about relative
        # magnitudes and projections, the specific 0-point (North vs East) cancels out
        # as long as it's consistent for both players.
        dir_p1 = np.radians(df["direction_p1"].fillna(0))
        dir_p2 = np.radians(df["direction_p2"].fillna(0))

        # Velocity Vectors
        vx_p1 = df["speed_p1"] * np.cos(dir_p1)
        vy_p1 = df["speed_p1"] * np.sin(dir_p1)
        vx_p2 = df["speed_p2"] * np.cos(dir_p2)
        vy_p2 = df["speed_p2"] * np.sin(dir_p2)

        # Acceleration Vectors
        # We use the acceleration magnitude directed along the motion direction
        # This is a standard approximation when orientation/jerk is not fully modeled
        ax_p1 = df["acceleration_p1"] * np.cos(dir_p1)
        ay_p1 = df["acceleration_p1"] * np.sin(dir_p1)
        ax_p2 = df["acceleration_p2"] * np.cos(dir_p2)
        ay_p2 = df["acceleration_p2"] * np.sin(dir_p2)

        # Relative Vectors (P1 - P2) or (P2 - P1)?
        # Distance is symmetric. Let's define vector from P2 to P1.
        rx = df["x_position_p1"] - df["x_position_p2"]
        ry = df["y_position_p1"] - df["y_position_p2"]

        rvx = vx_p1 - vx_p2
        rvy = vy_p1 - vy_p2

        rax = ax_p1 - ax_p2
        ray = ay_p1 - ay_p2

        # Current Distance (d0)
        d0 = np.sqrt(rx**2 + ry**2)

        # Unit Vector connecting P2 to P1
        # Handle division by zero (if d0 is 0, players are already in contact)
        # We replace 0 with epsilon to avoid NaNs
        d0_safe = np.where(d0 < 1e-6, 1e-6, d0)
        ux = rx / d0_safe
        uy = ry / d0_safe

        # Project Relative Velocity and Acceleration onto Unit Vector
        # v_rel: rate of change of distance (negative if closing)
        v_rel = rvx * ux + rvy * uy

        # a_rel: rate of change of closing speed
        a_rel = rax * ux + ray * uy

        return d0, v_rel, a_rel

    def _solve_quadratic_min(self, d0, v, a):
        """
        Finds the minimum value of d(t) = d0 + v*t + 0.5*a*t^2
        within the window t in [0, 1.0].
        """
        # Candidate times: 0.0, 1.0, and vertex t* = -v/a
        t_start = np.zeros_like(d0)
        t_end = np.ones_like(d0)

        # Calculate vertex
        # Handle a=0 case (linear motion)
        # If a ~ 0, set t_vertex to infinity (or a large number outside [0,1])
        a_safe = np.where(np.abs(a) < 1e-6, 1e-6, a)
        t_vertex = -v / a_safe

        # We only care about vertex if it falls within [0, 1]
        # We can clamp t_vertex to [0, 1] and evaluate.
        # If vertex is outside, the clamp will put it at 0 or 1, which we already check.
        t_vertex_clamped = np.clip(t_vertex, 0.0, 1.0)

        # Evaluate function at candidate times
        def eval_quad(t):
            return d0 + v * t + 0.5 * a * (t**2)

        d_start = eval_quad(t_start)
        d_end = eval_quad(t_end)
        d_vertex = eval_quad(t_vertex_clamped)

        # Take the minimum
        min_dist = np.minimum(d_start, np.minimum(d_end, d_vertex))

        # Note: If min_dist is negative, it implies the algebraic quadratic crossed zero.
        # Physically this means contact occurred (distance=0).
        # Since our check is min_dist < THRESHOLD (where THRESHOLD > 0),
        # negative values are correctly retained.

        return min_dist

    def apply_gating(self, df, load_cached_data=True):
        """
        Applies the gating logic to the dataframe.

        Args:
            df (pd.DataFrame): Merged dataframe containing tracking data.
            load_cached_data (bool): Whether to use caching.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        # 1. Cache Check
        # Generate a hash based on dataframe length and GATING_THRESHOLD to ensure validity
        cache_params = {
            "df_len": len(df),
            "threshold": GATING_THRESHOLD,
            "sentinel": SENTINEL_VALUE,
            "stage": "gated_data",
        }

        if load_cached_data:
            cached_df = self.cache_manager.load("gated_survivors", cache_params)
            if cached_df is not None:
                self.logger.info(
                    f"Loaded gated data from cache. Shape: {cached_df.shape}"
                )
                return cached_df

        self.logger.info(
            f"Applying Relaxed Quadratic Gating (Threshold={GATING_THRESHOLD} yds)..."
        )

        # 2. Separate Ground vs Player-Player
        # Ground interactions are identified by SENTINEL_VALUE in distance
        # (or nfl_player_id_2 == 'G', but distance is faster if already computed)
        # We rely on the sentinel value strategy defined in config/data_processing
        is_ground = df["distance"] == SENTINEL_VALUE

        df_ground = df[is_ground].copy()
        df_players = df[~is_ground].copy()

        self.logger.info(f"Ground interactions preserved: {len(df_ground)}")
        self.logger.info(f"Player-Player candidates before gating: {len(df_players)}")

        if len(df_players) > 0:
            # 3. Compute Kinematics for Players
            # Ensure no NaNs in critical columns (tracking data might be missing)
            # We filter out rows where we can't compute physics
            valid_tracking_mask = (
                df_players["x_position_p1"].notna()
                & df_players["x_position_p2"].notna()
                & df_players["speed_p1"].notna()
                & df_players["speed_p2"].notna()
            )

            df_valid = df_players[valid_tracking_mask].copy()

            d0, v_rel, a_rel = self._calculate_projected_kinematics(df_valid)

            # 4. Solve for Min Distance
            min_dist = self._solve_quadratic_min(d0, v_rel, a_rel)

            # 5. Apply Threshold
            # Keep if min_dist < Threshold
            keep_mask = min_dist < GATING_THRESHOLD

            df_survivors = df_valid[keep_mask].copy()

            # Add debugging info
            self.logger.info(
                f"Gating Survival Rate: {len(df_survivors)}/{len(df_players)} "
                f"({len(df_survivors)/len(df_players):.2%})"
            )
        else:
            df_survivors = pd.DataFrame(columns=df.columns)

        # 6. Recombine
        df_final = pd.concat([df_ground, df_survivors], axis=0, ignore_index=True)

        # 7. Save to Cache
        self.cache_manager.save(df_final, "gated_survivors", cache_params)

        self.logger.info(f"Final Gated Dataset Shape: {df_final.shape}")
        return df_final
