import pandas as pd
import numpy as np
import os
from library.config import ProjectConfig
from library.utils import get_logger

logger = get_logger("PhysicsEngine")


class PhysicsEngine:
    """
    Implements core mathematical transformations and physics calculations
    for the Physically-Consistent Hybrid-Context Dual-Stream GBDT.

    This engine provides stateless transformation functions to be used
    within the feature engineering pipeline.
    """

    @staticmethod
    def calculate_ego_dynamics(df_tracking: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates ego-centric dynamics: Surge, Sway, and their derivatives (Acceleration, Jerk).

        Logic:
            1. Projects velocity onto player orientation to get Surge (longitudinal) and Sway (lateral).
            2. Differentiates Velocity -> Acceleration.
            3. Differentiates Acceleration -> Jerk.

        Args:
            df_tracking (pd.DataFrame): Tracking data containing 'speed', 'direction', 'orientation',
                                        'game_play', 'nfl_player_id', 'step'.

        Returns:
            pd.DataFrame: Tracking data enriched with 'v_surge', 'v_sway', 'a_surge', 'a_sway', 'j_surge', 'j_sway'.
        """
        logger.info("Calculating Ego-Dynamics (Surge/Sway/Jerk)...")

        # Ensure data is sorted for temporal differentiation
        # We operate on a copy to avoid side effects
        df = df_tracking.sort_values(by=["game_play", "nfl_player_id", "step"]).copy()

        # Convert degrees to radians
        # NFL Data: 0-360 degrees.
        # We calculate the relative angle between motion (direction) and facing (orientation).
        dir_rad = np.radians(df["direction"].fillna(0))
        ori_rad = np.radians(df["orientation"].fillna(0))

        # Angular difference
        diff_rad = dir_rad - ori_rad

        # Velocity Components Projection
        # Surge: Component in the direction of orientation
        # Sway: Component perpendicular to orientation
        df["v_surge"] = df["speed"] * np.cos(diff_rad)
        df["v_sway"] = df["speed"] * np.sin(diff_rad)

        # Time delta (10Hz data)
        dt = 0.1

        # Define differentiation helper
        def calc_diff(x):
            return x.diff() / dt

        # Group by player within play to prevent diffing across boundaries
        # Using transform maintains the original index structure
        grp = df.groupby(["game_play", "nfl_player_id"])

        # 1st Derivative: Acceleration
        df["a_surge"] = grp["v_surge"].transform(calc_diff)
        df["a_sway"] = grp["v_sway"].transform(calc_diff)

        # 2nd Derivative: Jerk
        df["j_surge"] = grp["a_surge"].transform(calc_diff)
        df["j_sway"] = grp["a_sway"].transform(calc_diff)

        # Fill NaNs created by diff (first 2 steps of every play will be NaN)
        # We fill with 0.0 assuming static start or continuity
        cols_to_fill = ["a_surge", "a_sway", "j_surge", "j_sway"]
        df[cols_to_fill] = df[cols_to_fill].fillna(0.0)

        return df

    @staticmethod
    def calculate_relational_metrics(df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates relational metrics between two entities (Player-Player).
        Expects a dataframe with suffix '_p1' and '_p2' for coordinates.

        Args:
            df_merged (pd.DataFrame): Merged dataframe with x_position_p1, y_position_p1, etc.

        Returns:
            pd.DataFrame: Dataframe enriched with 'distance' and 'closure_rate'.
        """
        logger.info("Calculating Relational Metrics (Distance, Closure Rate)...")

        # Euclidean Distance
        dx = df_merged["x_position_p1"] - df_merged["x_position_p2"]
        dy = df_merged["y_position_p1"] - df_merged["y_position_p2"]
        df_merged["distance"] = np.sqrt(dx**2 + dy**2)

        # Closure Rate: - d(distance) / dt
        # Positive closure rate means distance is decreasing (players approaching)
        dt = 0.1

        # Determine grouping key for temporal differentiation
        if "contact_id" in df_merged.columns:
            # Optimal case: we have the unique contact identifier
            group_cols = ["contact_id"]
        else:
            # Fallback: group by play and pair
            group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

        # Ensure sorting for diff
        # Note: We assume the input df is generally ordered, but to be safe for diff:
        # (Sorting is expensive, so we rely on the pipeline to pass sorted data or accept minor overhead)
        # df_merged = df_merged.sort_values(by=group_cols + ["step"])

        # Calculate diff
        dist_diff = df_merged.groupby(group_cols)["distance"].diff()

        df_merged["closure_rate"] = -(dist_diff / dt)

        # Fill NaNs (start of play)
        df_merged["closure_rate"] = df_merged["closure_rate"].fillna(0.0)

        return df_merged

    @staticmethod
    def calculate_visual_metrics(
        df_base: pd.DataFrame, df_helmets: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates visual consensus metrics (IoU) for the pairs in df_base.
        Also calculates Visual Looming Rate.

        Args:
            df_base (pd.DataFrame): Dataframe containing 'game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2'.
            df_helmets (pd.DataFrame): Helmets dataframe with 'game_play', 'step', 'nfl_player_id', 'view', 'left', 'width', 'top', 'height'.

        Returns:
            pd.DataFrame: df_base enriched with IoU metrics and visual looming.
        """
        logger.info("Calculating Visual Metrics (IoU, Looming)...")

        # Optimization: Filter helmets to only plays present in df_base
        relevant_plays = df_base["game_play"].unique()
        df_h = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        # Calculate frame for df_base to align with helmets
        # Step is 10Hz, Video is 59.94Hz. Snap (Step 0) is at 5s (Frame ~300).
        df_base = df_base.copy()
        df_base["frame"] = (df_base["step"] * 5.994 + 300).round().astype(int)

        # Standardize IDs to strings to prevent merge errors (Cite debug_lesson_13)
        df_base["nfl_player_id_1"] = df_base["nfl_player_id_1"].astype(str)
        df_base["nfl_player_id_2"] = df_base["nfl_player_id_2"].astype(str)
        df_h["nfl_player_id"] = df_h["nfl_player_id"].astype(str)

        # Split helmets by view for efficient merging
        df_h_side = df_h[df_h["view"] == "Sideline"].copy()
        df_h_end = df_h[df_h["view"] == "Endzone"].copy()

        def merge_and_calc_iou(base, helmets, view_suffix):
            # 1. Merge P1 Helmet Box
            tmp = (
                pd.merge(
                    base,
                    helmets[
                        [
                            "game_play",
                            "frame",
                            "nfl_player_id",
                            "left",
                            "width",
                            "top",
                            "height",
                        ]
                    ],
                    left_on=["game_play", "frame", "nfl_player_id_1"],
                    right_on=["game_play", "frame", "nfl_player_id"],
                    how="left",
                )
                .rename(
                    columns={
                        "left": "x1_p1",
                        "width": "w_p1",
                        "top": "y1_p1",
                        "height": "h_p1",
                    }
                )
                .drop(columns=["nfl_player_id"])
            )

            # 2. Merge P2 Helmet Box
            tmp = (
                pd.merge(
                    tmp,
                    helmets[
                        [
                            "game_play",
                            "frame",
                            "nfl_player_id",
                            "left",
                            "width",
                            "top",
                            "height",
                        ]
                    ],
                    left_on=["game_play", "frame", "nfl_player_id_2"],
                    right_on=["game_play", "frame", "nfl_player_id"],
                    how="left",
                )
                .rename(
                    columns={
                        "left": "x1_p2",
                        "width": "w_p2",
                        "top": "y1_p2",
                        "height": "h_p2",
                    }
                )
                .drop(columns=["nfl_player_id"])
            )

            # 3. Vectorized IoU Calculation
            # P1
            x1_p1 = tmp["x1_p1"]
            y1_p1 = tmp["y1_p1"]
            x2_p1 = x1_p1 + tmp["w_p1"]
            y2_p1 = y1_p1 + tmp["h_p1"]
            area_p1 = tmp["w_p1"] * tmp["h_p1"]

            # P2
            x1_p2 = tmp["x1_p2"]
            y1_p2 = tmp["y1_p2"]
            x2_p2 = x1_p2 + tmp["w_p2"]
            y2_p2 = y1_p2 + tmp["h_p2"]
            area_p2 = tmp["w_p2"] * tmp["h_p2"]

            # Intersection
            xi1 = np.maximum(x1_p1, x1_p2)
            yi1 = np.maximum(y1_p1, y1_p2)
            xi2 = np.minimum(x2_p1, x2_p2)
            yi2 = np.minimum(y2_p1, y2_p2)

            inter_w = np.maximum(0, xi2 - xi1)
            inter_h = np.maximum(0, yi2 - yi1)
            inter_area = inter_w * inter_h

            # Union
            union_area = area_p1 + area_p2 - inter_area

            # IoU
            iou = np.where(union_area > 0, inter_area / union_area, 0.0)

            # Handle missing data (NaNs result in 0 IoU)
            iou = np.nan_to_num(iou, nan=0.0)

            return pd.Series(iou, index=base.index, name=f"iou_{view_suffix}")

        # Calculate for both views
        iou_side = merge_and_calc_iou(df_base, df_h_side, "sideline")
        iou_end = merge_and_calc_iou(df_base, df_h_end, "endzone")

        df_base["iou_sideline"] = iou_side
        df_base["iou_endzone"] = iou_end

        # Consensus Metrics
        df_base["max_iou"] = df_base[["iou_sideline", "iou_endzone"]].max(axis=1)
        df_base["min_iou"] = df_base[["iou_sideline", "iou_endzone"]].min(axis=1)
        df_base["iou_diff"] = (df_base["iou_sideline"] - df_base["iou_endzone"]).abs()

        # Visual Looming Rate: d(Max_IoU) / dt
        # Used to verify physical closure rate
        if "contact_id" in df_base.columns:
            group_cols = ["contact_id"]
        else:
            group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

        dt = 0.1
        df_base["visual_looming"] = df_base.groupby(group_cols)["max_iou"].diff() / dt
        df_base["visual_looming"] = df_base["visual_looming"].fillna(0.0)

        return df_base

    @staticmethod
    def create_lagged_features(
        df: pd.DataFrame, features: list, lags: list
    ) -> pd.DataFrame:
        """
        Creates lagged versions of specified features to flatten temporal context.

        Args:
            df (pd.DataFrame): Input dataframe.
            features (list): List of column names to lag.
            lags (list): List of integer lags (e.g., [-2, -1, 0, 1, 2]).
                         Negative lag usually implies past (t-k) in standard shift notation,
                         but here we interpret strictly as pandas shift(lag).
                         If lag=1, we get value from previous row.
                         If lag=-1, we get value from next row.

        Returns:
            pd.DataFrame: Dataframe with added lagged columns.
        """
        logger.info(
            f"Creating Lagged Features for {len(features)} features over {len(lags)} lags..."
        )

        # Determine grouping key
        if "contact_id" in df.columns:
            group_cols = ["contact_id"]
        elif "nfl_player_id_2" in df.columns:
            group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        else:
            # Single player case (Stream B)
            group_cols = ["game_play", "nfl_player_id"]

        grouper = df.groupby(group_cols)

        new_cols = {}

        for lag in lags:
            for feat in features:
                col_name = f"{feat}_lag_{lag}"

                if lag == 0:
                    # Lag 0 is the current value
                    new_cols[col_name] = df[feat]
                else:
                    # Use pandas shift
                    # shift(k) takes value from k steps before.
                    new_cols[col_name] = grouper[feat].shift(lag)

        # Create DataFrame from new columns
        df_lags = pd.DataFrame(new_cols, index=df.index)

        # Fill missing values (boundaries) with sentinel
        df_lags = df_lags.fillna(ProjectConfig.MISSING_SENTINEL)

        # Concatenate with original dataframe
        # Note: We keep original columns as well, though lag_0 might duplicate them.
        # The downstream model selector will pick what it needs.
        df_out = pd.concat([df, df_lags], axis=1)

        # Deduplicate columns if any (e.g. if lag 0 overwrote original name, though here we used suffix)
        df_out = df_out.loc[:, ~df_out.columns.duplicated()]

        return df_out

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_cols: list):
        """
        Validates that the dataframe contains required columns.
        Raises RuntimeError if validation fails to ensure pipeline integrity.
        """
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Schema Validation Failed. Missing columns: {missing}")
        return True
