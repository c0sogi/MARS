import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import reduce_mem_usage, setup_logger
from library.data_loader import DataLoader


class HelmetFeatureEngineer:
    """
    Implements the visual-geometric feature generation pipeline (Stream B).
    Processes helmet bounding boxes to generate IoU, Distance, and Agitation features.
    """

    def __init__(self):
        self.config = Config
        self.logger = setup_logger("HelmetFE")
        self.data_loader = DataLoader()

    def calculate_iou(self, df):
        """
        Vectorized Intersection over Union calculation.
        Expects columns: left_p1, width_p1, top_p1, height_p1, and same for _p2.
        """
        # Coordinates P1
        x1_p1 = df["left_p1"]
        x2_p1 = df["left_p1"] + df["width_p1"]
        y1_p1 = df["top_p1"]
        y2_p1 = df["top_p1"] + df["height_p1"]
        area_p1 = df["width_p1"] * df["height_p1"]

        # Coordinates P2
        x1_p2 = df["left_p2"]
        x2_p2 = df["left_p2"] + df["width_p2"]
        y1_p2 = df["top_p2"]
        y2_p2 = df["top_p2"] + df["height_p2"]
        area_p2 = df["width_p2"] * df["height_p2"]

        # Intersection
        x1_i = np.maximum(x1_p1, x1_p2)
        y1_i = np.maximum(y1_p1, y1_p2)
        x2_i = np.minimum(x2_p1, x2_p2)
        y2_i = np.minimum(y2_p1, y2_p2)

        w_i = np.maximum(0, x2_i - x1_i)
        h_i = np.maximum(0, y2_i - y1_i)
        intersection = w_i * h_i

        # Union
        union = area_p1 + area_p2 - intersection

        # IoU (handle divide by zero)
        iou = intersection / (union + 1e-6)

        # If P2 is missing (NaN), IoU is 0
        return iou.fillna(0)

    def calculate_centroid_distance(self, df):
        """
        Vectorized Centroid Distance calculation.
        Expects center_x_p1, center_y_p1, etc.
        """
        dx = df["center_x_p1"] - df["center_x_p2"]
        dy = df["center_y_p1"] - df["center_y_p2"]
        dist = np.sqrt(dx**2 + dy**2)

        # If P2 is missing (e.g. Ground), fill with -1 or large value.
        # Using -1 allows trees to split easily.
        return dist.fillna(-1)

    def preprocess_helmets(self, df_helmets):
        """
        Computes intrinsic helmet features (Center, Area, Speed, Agitation).
        """
        self.logger.info("Preprocessing raw helmet data...")

        # Ensure sorted for temporal diffs
        df_helmets = df_helmets.sort_values(
            ["game_play", "view", "nfl_player_id", "frame"]
        )

        # Geometric centers
        df_helmets["center_x"] = df_helmets["left"] + df_helmets["width"] / 2
        df_helmets["center_y"] = df_helmets["top"] + df_helmets["height"] / 2
        df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

        # Kinematics (Pixel Speed & Agitation)
        # Group by unique entity in video
        grouper = df_helmets.groupby(["game_play", "view", "nfl_player_id"])

        # Shift 1 frame
        prev_cx = grouper["center_x"].shift(1)
        prev_cy = grouper["center_y"].shift(1)

        # Pixel Speed
        df_helmets["h_speed"] = np.sqrt(
            (df_helmets["center_x"] - prev_cx) ** 2
            + (df_helmets["center_y"] - prev_cy) ** 2
        )

        # Agitation (Acceleration magnitude / Jerk proxy)
        prev_speed = grouper["h_speed"].shift(1)
        df_helmets["h_agitation"] = (df_helmets["h_speed"] - prev_speed).abs()

        # Fill NaNs (first frames)
        df_helmets = df_helmets.fillna(0)

        return reduce_mem_usage(df_helmets)

    def process_view(self, metadata_df, helmets_df, view_name):
        """
        Merges metadata with helmet data for a specific view and calculates pair features.
        """
        self.logger.info(f"Processing view: {view_name}")

        # Filter helmets for this view
        view_helmets = helmets_df[helmets_df["view"] == view_name].copy()

        # Prepare Metadata for merge
        # We need to map step -> frame.
        # Frame = 300 + round(step * 5.994)
        # We do this mapping in the main loop, but here we expect metadata to have 'frame_approx'

        # Merge Player 1
        # Rename helmet cols to _p1
        h_p1 = view_helmets.add_suffix("_p1")
        # Restore join keys
        h_p1 = h_p1.rename(
            columns={
                "game_play_p1": "game_play",
                "nfl_player_id_p1": "nfl_player_id_1",
                "frame_p1": "frame_approx",
            }
        )

        # Left join to keep all labels
        merged = pd.merge(
            metadata_df,
            h_p1,
            on=["game_play", "nfl_player_id_1", "frame_approx"],
            how="left",
        )

        # Merge Player 2
        # Rename helmet cols to _p2
        h_p2 = view_helmets.add_suffix("_p2")
        h_p2 = h_p2.rename(
            columns={
                "game_play_p2": "game_play",
                "nfl_player_id_p2": "nfl_player_id_2",
                "frame_p2": "frame_approx",
            }
        )

        merged = pd.merge(
            merged,
            h_p2,
            on=["game_play", "nfl_player_id_2", "frame_approx"],
            how="left",
        )

        # Calculate Pair Features
        merged[f"h_iou_{view_name}"] = self.calculate_iou(merged)
        merged[f"h_dist_{view_name}"] = self.calculate_centroid_distance(merged)

        # Keep relevant single player features (Agitation, Area)
        # We rename them to include view
        rename_map = {
            "h_agitation_p1": f"h_agitation_p1_{view_name}",
            "h_agitation_p2": f"h_agitation_p2_{view_name}",
            "area_p1": f"h_area_p1_{view_name}",
            "area_p2": f"h_area_p2_{view_name}",
        }
        merged = merged.rename(columns=rename_map)

        # Select only the new columns to return
        cols_to_keep = [f"h_iou_{view_name}", f"h_dist_{view_name}"] + list(
            rename_map.values()
        )

        # Fill NaNs for features (missing helmets)
        # IoU -> 0, Dist -> -1, others -> 0
        merged[f"h_iou_{view_name}"] = merged[f"h_iou_{view_name}"].fillna(0)
        merged[f"h_dist_{view_name}"] = merged[f"h_dist_{view_name}"].fillna(-1)
        merged[list(rename_map.values())] = merged[list(rename_map.values())].fillna(0)

        return merged[cols_to_keep]

    def create_window_features(self, df):
        """
        Generates temporal lag features for the geometric signals.
        """
        self.logger.info("Generating temporal window features for helmets...")

        # Sort for shifting
        df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

        # Features to lag
        cols_to_lag = ["h_iou", "h_dist", "h_agitation_p1", "h_agitation_p2"]
        shifts = [-4, -2, -1, 1, 2, 4]  # Micro-window

        grouper = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        for col in cols_to_lag:
            if col not in df.columns:
                continue
            for s in shifts:
                df[f"{col}_shift_{s}"] = grouper[col].shift(s)

        return df.fillna(0)

    def create_features(
        self, metadata_df, helmets_df=None, mode="train", load_cached_data=True
    ):
        """
        Main pipeline execution method.
        """
        cache_path = os.path.join(
            self.config.WORKING_DIR, f"{mode}_helmet_features.parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(f"Loading cached helmet features from {cache_path}")
            return pd.read_parquet(cache_path)

        self.logger.info(f"Generating helmet features for {mode} set...")

        # 1. Load Helmet Data
        if helmets_df is None:
            train_helmets, test_helmets = self.data_loader.load_helmet_data(
                load_cached_data=True
            )
            helmets_df = test_helmets if mode == "test" else train_helmets

            # Cleanup
            if mode == "test":
                del train_helmets
            else:
                del test_helmets
            gc.collect()

        # 2. Filter Helmets (Optimization)
        relevant_plays = metadata_df["game_play"].unique()
        helmets_df = helmets_df[helmets_df["game_play"].isin(relevant_plays)].copy()

        if helmets_df.empty:
            raise ValueError(f"No helmet data found for {mode} plays.")

        # 3. Preprocess Helmets (Kinematics)
        helmets_df = self.preprocess_helmets(helmets_df)

        # 4. Map Steps to Frames
        # Frame 300 is roughly 5.0s (Step 0)
        # 59.94 Hz vs 10 Hz -> ~5.994 frames per step
        self.logger.info("Mapping steps to video frames...")
        metadata_df["frame_approx"] = (
            (300 + metadata_df["step"] * 5.994).round().astype(int)
        )

        # 5. Process Views
        # Sideline
        feats_sideline = self.process_view(metadata_df, helmets_df, "Sideline")
        # Endzone
        feats_endzone = self.process_view(metadata_df, helmets_df, "Endzone")

        # 6. Aggregate Views
        self.logger.info("Aggregating views (Late Fusion)...")

        # Concatenate features to metadata
        # Since process_view returns df aligned with metadata (if index preserved? No, merge shuffles)
        # We must be careful. process_view performed a merge on metadata.
        # The safest way is to have process_view return the full merged df, or concat carefully.
        # Let's modify process_view logic slightly in thought?
        # Actually, process_view does a left join on metadata. The result has same length as metadata IF metadata is unique on keys.
        # Metadata is unique on contact_id, but keys used were (game_play, p1, frame).
        # Multiple steps might map to same frame? (Rounding). Yes.
        # But (game_play, p1, p2, step) is unique.
        # Let's perform the merge inside this function to be safe.

        # Re-implementation of Merge Logic here for safety:
        # Sideline
        sideline_full = self.process_view(metadata_df, helmets_df, "Sideline")
        # Endzone
        endzone_full = self.process_view(metadata_df, helmets_df, "Endzone")

        # Since process_view returns rows matching metadata_df *in order*?
        # No, pd.merge might change order.
        # We should merge the results back to metadata_df based on index or keys.
        # But process_view returns only feature columns.
        # Let's assume process_view returns a dataframe that we can concat if we ensure order.
        # Better: process_view should return the full dataframe with keys.

        # Correct approach:
        # We will merge features into metadata_df one by one.

        # Helper to get view features with keys
        def get_view_df(view_name):
            view_h = helmets_df[helmets_df["view"] == view_name].copy()

            # P1
            h_p1 = view_h.add_suffix("_p1").rename(
                columns={
                    "game_play_p1": "game_play",
                    "nfl_player_id_p1": "nfl_player_id_1",
                    "frame_p1": "frame_approx",
                }
            )
            # P2
            h_p2 = view_h.add_suffix("_p2").rename(
                columns={
                    "game_play_p2": "game_play",
                    "nfl_player_id_p2": "nfl_player_id_2",
                    "frame_p2": "frame_approx",
                }
            )

            # Merge P1
            res = pd.merge(
                metadata_df,
                h_p1,
                on=["game_play", "nfl_player_id_1", "frame_approx"],
                how="left",
            )
            # Merge P2
            res = pd.merge(
                res,
                h_p2,
                on=["game_play", "nfl_player_id_2", "frame_approx"],
                how="left",
            )

            # Calc
            res[f"h_iou_{view_name}"] = self.calculate_iou(res)
            res[f"h_dist_{view_name}"] = self.calculate_centroid_distance(res)

            # Rename agitations/areas
            rename_map = {
                "h_agitation_p1": f"h_agitation_p1_{view_name}",
                "h_agitation_p2": f"h_agitation_p2_{view_name}",
                "area_p1": f"h_area_p1_{view_name}",
                "area_p2": f"h_area_p2_{view_name}",
            }
            res = res.rename(columns=rename_map)

            # Fill NaNs
            res[f"h_iou_{view_name}"] = res[f"h_iou_{view_name}"].fillna(0)
            res[f"h_dist_{view_name}"] = res[f"h_dist_{view_name}"].fillna(-1)
            for col in rename_map.values():
                res[col] = res[col].fillna(0)

            return res[
                ["contact_id", f"h_iou_{view_name}", f"h_dist_{view_name}"]
                + list(rename_map.values())
            ]

        df_sideline = get_view_df("Sideline")
        df_endzone = get_view_df("Endzone")

        # Merge back to metadata
        final_df = pd.merge(metadata_df, df_sideline, on="contact_id", how="left")
        final_df = pd.merge(final_df, df_endzone, on="contact_id", how="left")

        # 7. Compute Aggregated Features
        # Max IoU
        final_df["h_iou"] = np.maximum(
            final_df["h_iou_Sideline"], final_df["h_iou_Endzone"]
        )

        # Min Distance (use 9999 for -1 during min calc to avoid selecting missing view, then revert?)
        # Logic: If one view is missing (-1), we want the other view. If both missing, -1.
        # If both present, min.
        d_s = final_df["h_dist_Sideline"].replace(-1, 9999)
        d_e = final_df["h_dist_Endzone"].replace(-1, 9999)
        min_d = np.minimum(d_s, d_e)
        final_df["h_dist"] = min_d.replace(9999, -1)

        # Max Agitation/Area
        final_df["h_agitation_p1"] = np.maximum(
            final_df["h_agitation_p1_Sideline"], final_df["h_agitation_p1_Endzone"]
        )
        final_df["h_agitation_p2"] = np.maximum(
            final_df["h_agitation_p2_Sideline"], final_df["h_agitation_p2_Endzone"]
        )
        final_df["h_area_p1"] = np.maximum(
            final_df["h_area_p1_Sideline"], final_df["h_area_p1_Endzone"]
        )

        # 8. Window Features
        final_df = self.create_window_features(final_df)

        # 9. Save
        # Drop intermediate view columns to save space?
        # Maybe keep them, tree models might find specific view useful.
        # But for "Stream B" we want a consolidated representation. Let's drop to keep file size manageable.
        drop_cols = [c for c in final_df.columns if "Sideline" in c or "Endzone" in c]
        final_df = final_df.drop(columns=drop_cols)

        self.logger.info(f"Saving features to {cache_path}...")
        final_df = reduce_mem_usage(final_df)
        final_df.to_parquet(cache_path, index=False)

        return final_df
