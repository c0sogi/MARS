import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import reduce_mem_usage
from library.feature_shared import (
    calculate_euclidean_distance,
    calculate_closure_rate,
    create_temporal_lags,
)
from library.data_manager import DataManager


class StreamAFeatureGenerator:
    """
    Implements the Relational-Visual Pipeline for Player-Player interactions.
    Generates features for Stream A:
    - Relational Scalars (Distance, Closure Rate, Relative Speed)
    - Visual Temporal Pyramids (IoU, Box Distance from Sideline/Endzone)
    - Exponential Temporal Lags
    """

    def __init__(self):
        self.config = Config
        self.data_manager = DataManager()
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def generate_features(self, mode="train", load_cached_data=True):
        """
        Generates or loads Stream A features.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            pd.DataFrame: Feature matrix for Stream A.
        """
        cache_file = os.path.join(self.working_dir, f"features_stream_a_{mode}.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"[Stream A] Loading cached features from {cache_file}...")
            return pd.read_parquet(cache_file)

        print(f"[Stream A] Generating features for {mode}...")

        # 1. Load Merged Data
        df = self.data_manager.load_dataset(mode=mode, load_cached_data=True)

        # 2. Filter for Player-Player Interactions Only
        # Stream A is strictly for P-P. P-G is handled by Stream B.
        print(
            f"[Stream A] Filtering for Player-Player interactions (Total rows before: {len(df)})..."
        )
        df = df[df["nfl_player_id_2"] != "G"].copy()
        print(f"[Stream A] Rows after filtering: {len(df)}")

        if len(df) == 0:
            print("[Stream A] Warning: No Player-Player interactions found.")
            return pd.DataFrame()

        # 3. Tracking Feature Engineering (Relational Scalars)
        print("[Stream A] Computing Relational Scalars...")

        # Euclidean Distance
        df["dist_p1_p2"] = calculate_euclidean_distance(
            df["x_position_p1"],
            df["y_position_p1"],
            df["x_position_p2"],
            df["y_position_p2"],
        )

        # Closure Rate
        # Must group by pair to calculate diff correctly
        # Sort by game_play, pair, step to ensure time order
        df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
        )

        # We can apply closure rate calculation per group
        # Using a lambda is slow, let's use the shared function with a groupby transform if possible,
        # or just apply it carefully. The shared function expects a series.
        # We'll use groupby().transform()
        df["closure_rate"] = df.groupby(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        )["dist_p1_p2"].transform(lambda x: calculate_closure_rate(x))

        # Relative Speed (Magnitude of velocity difference vector)
        # Convert speed/direction to components
        # Direction: 0=North(Y), 90=East(X).
        # vx = speed * sin(rad), vy = speed * cos(rad)

        for p in ["p1", "p2"]:
            rad = np.radians(df[f"direction_{p}"].fillna(0))
            df[f"v_x_{p}"] = df[f"speed_{p}"] * np.sin(rad)
            df[f"v_y_{p}"] = df[f"speed_{p}"] * np.cos(rad)

        df["rel_speed"] = np.sqrt(
            (df["v_x_p1"] - df["v_x_p2"]) ** 2 + (df["v_y_p1"] - df["v_y_p2"]) ** 2
        )

        # Cyclical Encoding
        for col in ["direction_p1", "orientation_p1", "direction_p2", "orientation_p2"]:
            rad = np.radians(df[col].fillna(0))
            df[f"{col}_sin"] = np.sin(rad)
            df[f"{col}_cos"] = np.cos(rad)

        # 4. Visual Feature Engineering (Visual Pyramids)
        if self.config.FEATURE_CONFIG["stream_a"]["use_visuals"]:
            print("[Stream A] Computing Visual Features...")
            df = self._add_visual_features(df, mode)

        # 5. Temporal Lags
        print("[Stream A] Applying Temporal Lags...")

        # Define lag groups
        group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

        # Tracking Lags
        tracking_feats = [
            "dist_p1_p2",
            "closure_rate",
            "rel_speed",
            "speed_p1",
            "speed_p2",
            "acceleration_p1",
            "acceleration_p2",
            "direction_p1_sin",
            "direction_p1_cos",
            "orientation_p1_sin",
            "orientation_p1_cos",
        ]
        tracking_lags = self.config.FEATURE_CONFIG["tracking_lags"]

        df_tracking_lags = create_temporal_lags(
            df, group_cols, tracking_feats, tracking_lags
        )
        df = pd.concat([df, df_tracking_lags], axis=1)

        # Visual Lags (if exist)
        visual_feats = ["iou_sideline", "dist_sideline", "iou_endzone", "dist_endzone"]
        visual_lags = self.config.FEATURE_CONFIG["visual_lags"]

        if all(f in df.columns for f in visual_feats):
            df_visual_lags = create_temporal_lags(
                df, group_cols, visual_feats, visual_lags
            )
            df = pd.concat([df, df_visual_lags], axis=1)

        # 6. Cleanup and Save
        # Select features to keep
        # We keep identifiers, target, base features, and lag features
        # Drop intermediate calculation columns like v_x_p1, etc.
        drop_cols = ["v_x_p1", "v_y_p1", "v_x_p2", "v_y_p2"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        df = reduce_mem_usage(df)

        print(f"[Stream A] Saving {len(df)} rows to {cache_file}...")
        df.to_parquet(cache_file, index=False)

        return df

    def _add_visual_features(self, df, mode):
        """
        Helper to merge helmet data and calculate IoU/Box Distance.
        """
        # Load Helmets
        df_helmets = self.data_manager.load_helmets(mode)

        # Filter helmets to relevant plays
        relevant_plays = df["game_play"].unique()
        df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        # Map Step (10Hz) to Frame (~59.94Hz)
        # Snap is at frame 300 (5s). Step 0 is at snap.
        # Frame = 300 + step * (59.94 / 10) approx 300 + step * 6
        # Cast step to int32 to prevent OverflowError if step was downcasted to int8 (Cite debug_lesson_2)
        df["frame_approx"] = (300 + df["step"].astype(np.int32) * 6).astype(int)

        # Prepare Helmets for Merge
        # We need to merge twice per view: once for P1, once for P2

        views = ["Sideline", "Endzone"]

        for view in views:
            print(f"  - Processing {view} view...")

            # Filter helmets for this view
            helmets_view = df_helmets[df_helmets["view"] == view].copy()

            # Ensure merge keys match types
            helmets_view["nfl_player_id"] = helmets_view["nfl_player_id"].astype(str)

            # Merge P1
            df = pd.merge(
                df,
                helmets_view[
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
                left_on=["game_play", "frame_approx", "nfl_player_id_1"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
                suffixes=("", "_p1_vis"),
            )
            # Rename P1 columns explicitly if suffixes didn't catch (first merge usually keeps original names)
            rename_map_p1 = {
                "left": "left_p1",
                "width": "width_p1",
                "top": "top_p1",
                "height": "height_p1",
            }
            df.rename(columns=rename_map_p1, inplace=True)

            # Merge P2
            df = pd.merge(
                df,
                helmets_view[
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
                left_on=["game_play", "frame_approx", "nfl_player_id_2"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
                suffixes=("", "_p2_vis"),
            )
            # Rename P2 columns
            rename_map_p2 = {
                "left": "left_p2",
                "width": "width_p2",
                "top": "top_p2",
                "height": "height_p2",
            }
            df.rename(columns=rename_map_p2, inplace=True)

            # Drop redundant merge cols
            drop_merge_cols = [
                "frame",
                "nfl_player_id",
                "frame_p2_vis",
                "nfl_player_id_p2_vis",
            ]
            df.drop(
                columns=[c for c in drop_merge_cols if c in df.columns], inplace=True
            )

            # Calculate IoU and Distance
            # Fill missing boxes with sentinel -999 to allow calculation (will result in nonsense that tree can split on)
            # Actually, better to fill with 0 for IoU calculation logic, then mask result?
            # Or just calculate on valid rows.
            # Vectorized calculation:

            # Coordinates
            # P1
            x1_p1 = df["left_p1"]
            y1_p1 = df["top_p1"]
            x2_p1 = df["left_p1"] + df["width_p1"]
            y2_p1 = df["top_p1"] + df["height_p1"]

            # P2
            x1_p2 = df["left_p2"]
            y1_p2 = df["top_p2"]
            x2_p2 = df["left_p2"] + df["width_p2"]
            y2_p2 = df["top_p2"] + df["height_p2"]

            # Intersection
            xi1 = np.maximum(x1_p1, x1_p2)
            yi1 = np.maximum(y1_p1, y1_p2)
            xi2 = np.minimum(x2_p1, x2_p2)
            yi2 = np.minimum(y2_p1, y2_p2)

            inter_width = np.maximum(0, xi2 - xi1)
            inter_height = np.maximum(0, yi2 - yi1)
            inter_area = inter_width * inter_height

            # Union
            box1_area = df["width_p1"] * df["height_p1"]
            box2_area = df["width_p2"] * df["height_p2"]
            union_area = box1_area + box2_area - inter_area

            # IoU
            iou_col = f"iou_{view.lower()}"
            df[iou_col] = inter_area / union_area

            # Centroid Distance
            cx_p1 = x1_p1 + df["width_p1"] / 2
            cy_p1 = y1_p1 + df["height_p1"] / 2
            cx_p2 = x1_p2 + df["width_p2"] / 2
            cy_p2 = y1_p2 + df["height_p2"] / 2

            dist_col = f"dist_{view.lower()}"
            df[dist_col] = np.sqrt((cx_p1 - cx_p2) ** 2 + (cy_p1 - cy_p2) ** 2)

            # Fill NaNs (where one or both players weren't found) with Sentinel
            df[iou_col] = df[iou_col].fillna(-999)
            df[dist_col] = df[dist_col].fillna(-999)

            # Drop raw box columns to save memory
            box_cols = [
                "left_p1",
                "width_p1",
                "top_p1",
                "height_p1",
                "left_p2",
                "width_p2",
                "top_p2",
                "height_p2",
            ]
            df.drop(columns=box_cols, inplace=True)

        return df
