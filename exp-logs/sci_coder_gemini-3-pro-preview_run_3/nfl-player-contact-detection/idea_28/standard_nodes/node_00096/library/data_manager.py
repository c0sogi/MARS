import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    METADATA_PATHS,
    TRACKING_FILE_MAP,
    HELMET_FILE_MAP,
    WORKING_DIR,
    REQUIRED_TRACKING_COLS,
    SEED,
)
from library.utils import reduce_mem_usage, verify_schema, compute_config_hash
from library.physics_engine import calculate_iou_metrics

CONFIG_HASH = compute_config_hash()


class DataManager:
    def __init__(self, mode="train", debug=False):
        self.mode = mode
        self.debug = debug
        self.cache_dir = WORKING_DIR

        # Resolve paths based on mode
        self.metadata_path = METADATA_PATHS[mode]
        self.tracking_path = TRACKING_FILE_MAP[mode]
        self.helmet_path = HELMET_FILE_MAP[mode]

    def load_data(self, load_cached=True):
        """
        Main entry point to load and process data.
        Returns tuple: (df_stream_a, df_stream_b)
        """
        # Define cache paths
        # Cite solution_lesson_node_00093: Use config hash to prevent stale data loading
        cache_path_a = os.path.join(
            self.cache_dir, f"merged_stream_a_{self.mode}_{CONFIG_HASH}.parquet"
        )
        cache_path_b = os.path.join(
            self.cache_dir, f"merged_stream_b_{self.mode}_{CONFIG_HASH}.parquet"
        )

        # Attempt to load from cache
        if (
            load_cached
            and os.path.exists(cache_path_a)
            and os.path.exists(cache_path_b)
        ):
            print(f"Loading cached data for mode '{self.mode}'...")
            df_a = pd.read_parquet(cache_path_a)
            df_b = pd.read_parquet(cache_path_b)
            return df_a, df_b

        print(f"Processing data from scratch for mode '{self.mode}'...")

        # 1. Load Metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df_meta = pd.read_csv(self.metadata_path)
        if self.debug:
            # Sample metadata for debugging
            df_meta = df_meta.sample(
                n=min(5000, len(df_meta)), random_state=SEED
            ).copy()

        # 2. Load Raw Data (Tracking & Helmets)
        # Filter raw data to only include game_plays present in metadata to save memory
        relevant_game_plays = df_meta["game_play"].unique()

        print(f"Loading and filtering tracking data from {self.tracking_path}...")
        df_tracking = pd.read_csv(self.tracking_path)
        df_tracking = df_tracking[
            df_tracking["game_play"].isin(relevant_game_plays)
        ].copy()
        df_tracking = reduce_mem_usage(df_tracking)

        print(f"Loading and filtering helmet data from {self.helmet_path}...")
        df_helmets = pd.read_csv(self.helmet_path)
        df_helmets = df_helmets[
            df_helmets["game_play"].isin(relevant_game_plays)
        ].copy()
        df_helmets = reduce_mem_usage(df_helmets)

        # 3. Split Metadata into Streams
        # Stream A: Player-Player (nfl_player_id_2 != 'G')
        # Stream B: Player-Ground (nfl_player_id_2 == 'G')

        # Ensure IDs are strings for consistent comparison initially
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)

        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_meta_b = df_meta[mask_ground].copy()
        df_meta_a = df_meta[~mask_ground].copy()

        # 4. Process Stream A
        print("Processing Stream A (Interaction)...")
        df_a = self._process_stream_a(df_meta_a, df_tracking, df_helmets)

        # 5. Process Stream B
        print("Processing Stream B (Impact)...")
        df_b = self._process_stream_b(df_meta_b, df_tracking)

        # 6. Save to Cache
        print("Saving to cache...")
        os.makedirs(self.cache_dir, exist_ok=True)
        df_a.to_parquet(cache_path_a, index=False)
        df_b.to_parquet(cache_path_b, index=False)

        # Cleanup
        del df_tracking, df_helmets, df_meta
        gc.collect()

        return df_a, df_b

    def _process_stream_a(self, df_meta, df_tracking, df_helmets):
        if df_meta.empty:
            return pd.DataFrame()

        # --- Merge Tracking Data ---
        # We need tracking for P1 and P2

        # Prepare tracking data
        df_track_prep = df_tracking[REQUIRED_TRACKING_COLS].copy()

        # Ensure nfl_player_id_1 is int for merging
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

        # Merge P1
        df_merged = pd.merge(
            df_meta,
            df_track_prep.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="inner",  # Essential features
        )

        # Merge P2
        # Convert nfl_player_id_2 to int for merging (it was string 'G' safe before split)
        df_merged["nfl_player_id_2_int"] = (
            df_merged["nfl_player_id_2"].astype(float).astype(int)
        )

        df_merged = pd.merge(
            df_merged,
            df_track_prep.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2_int"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="inner",
        )

        # --- Merge Helmet Data & Calc IoU ---
        # Map step to frame: frame = 300 + round(step * 5.994)
        # 59.94 Hz video. 10 Hz tracking.
        # step 0 -> 0s -> frame 300 (snap)
        df_merged["frame_approx"] = (
            (300 + df_merged["step"] * 5.994).round().astype(int)
        )

        # Helper to merge specific view and calc IoU
        def merge_view_iou(df, view_name, col_prefix):
            # Filter helmets for view
            helmets_view = df_helmets[df_helmets["view"] == view_name].copy()

            # Select columns
            h_cols = [
                "game_play",
                "frame",
                "nfl_player_id",
                "left",
                "width",
                "top",
                "height",
            ]
            helmets_view = helmets_view[h_cols]

            # Merge P1 Helmets
            df = pd.merge(
                df,
                helmets_view.add_suffix("_p1"),
                left_on=["game_play", "frame_approx", "nfl_player_id_1"],
                right_on=["game_play_p1", "frame_p1", "nfl_player_id_p1"],
                how="left",
            )

            # Merge P2 Helmets
            df = pd.merge(
                df,
                helmets_view.add_suffix("_p2"),
                left_on=["game_play", "frame_approx", "nfl_player_id_2_int"],
                right_on=["game_play_p2", "frame_p2", "nfl_player_id_p2"],
                how="left",
            )

            # Calculate IoU
            box_cols = ["left", "width", "top", "height"]
            p1_cols = [c + "_p1" for c in box_cols]
            p2_cols = [c + "_p2" for c in box_cols]

            # FillNa with 0 (missing helmet -> 0 IoU)
            for c in p1_cols + p2_cols:
                if c in df.columns:
                    df[c] = df[c].fillna(0)
                else:
                    # Should not happen if merge worked, but safety
                    df[c] = 0

            # Vectorized IoU
            boxes1 = df[p1_cols].values
            boxes2 = df[p2_cols].values

            ious = calculate_iou_metrics(boxes1, boxes2)
            df[col_prefix] = ious

            # Drop temp columns
            drop_cols = (
                p1_cols
                + p2_cols
                + [
                    "game_play_p1",
                    "frame_p1",
                    "nfl_player_id_p1",
                    "game_play_p2",
                    "frame_p2",
                    "nfl_player_id_p2",
                ]
            )
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

            return df

        # Sideline IoU
        df_merged = merge_view_iou(df_merged, "Sideline", "iou_sideline")
        # Endzone IoU
        df_merged = merge_view_iou(df_merged, "Endzone", "iou_endzone")

        # Calculate Aggregates
        df_merged["iou_max"] = df_merged[["iou_sideline", "iou_endzone"]].max(axis=1)
        df_merged["iou_min"] = df_merged[["iou_sideline", "iou_endzone"]].min(axis=1)
        df_merged["iou_diff"] = (
            df_merged["iou_sideline"] - df_merged["iou_endzone"]
        ).abs()

        return reduce_mem_usage(df_merged)

    def _process_stream_b(self, df_meta, df_tracking):
        if df_meta.empty:
            return pd.DataFrame()

        # Stream B only needs P1 tracking
        # P2 is Ground, no tracking, no helmets

        df_track_prep = df_tracking[REQUIRED_TRACKING_COLS].copy()

        # Ensure nfl_player_id_1 is int
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

        # Merge P1
        # Direct merge, no suffix, keeping original tracking column names (x_position, speed, etc.)
        df_merged = pd.merge(
            df_meta,
            df_track_prep,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="inner",
        )

        return reduce_mem_usage(df_merged)
