import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import get_cache_path, save_to_cache, load_from_cache, set_seed


class DataManager:
    """
    Manages data loading, merging, and caching for the football contact detection task.
    Handles the alignment of tracking and helmet data with contact labels.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def load_data(self, split="train", load_cached_data=True):
        """
        Loads the dataset for the specified split, merging metadata, tracking, and helmet data.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The merged dataframe containing labels, tracking, and helmet info.
        """
        # 1. Construct Cache Key/Path
        # Include configuration parameters in the hash to ensure cache validity
        cache_config = {
            "split": split,
            "debug": Config.DEBUG,
            "tracking_cols": Config.TRACKING_BASE_COLS,
            "visual_cols": Config.VISUAL_BASE_COLS,
            "step_to_frame_formula": "300 + step * 6",
        }

        cache_path = get_cache_path(
            self.working_dir, f"merged_data_{split}", cache_config, "parquet"
        )

        # 2. Try Loading from Cache
        if load_cached_data:
            print(f"Attempting to load {split} data from cache: {cache_path}")
            df = load_from_cache(cache_path)
            if df is not None:
                print("Cache hit. Data loaded successfully.")
                return df
            print("Cache miss or file not found.")

        # 3. Load and Process from Scratch
        print(f"Processing {split} data from scratch...")

        # A. Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_META_PATH
            is_train_source = True
        elif split == "validation":
            meta_path = Config.VAL_META_PATH
            is_train_source = True
        elif split == "test":
            meta_path = Config.TEST_META_PATH
            is_train_source = False
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Debug Sampling: Reduce dataset size for faster iteration
        if Config.DEBUG:
            print("DEBUG Mode: Sampling data...")
            unique_games = df_meta["game_play"].unique()
            # Take first 2 games for debug
            sample_games = unique_games[:2] if len(unique_games) > 2 else unique_games
            df_meta = df_meta[df_meta["game_play"].isin(sample_games)].copy()
            print(
                f"Debug: Reduced metadata to {len(df_meta)} rows ({len(sample_games)} games)."
            )

        # B. Load Raw Data (Tracking & Helmets)
        if is_train_source:
            tracking_path = Config.TRAIN_TRACKING_PATH
            helmets_path = Config.TRAIN_HELMETS_PATH
        else:
            tracking_path = Config.TEST_TRACKING_PATH
            helmets_path = Config.TEST_HELMETS_PATH

        print(f"Loading tracking data from {tracking_path}...")
        df_tracking = pd.read_csv(tracking_path)

        print(f"Loading helmets data from {helmets_path}...")
        df_helmets = pd.read_csv(helmets_path)

        # C. Filter Raw Data (Optimization)
        # Only keep data relevant to the games in metadata to save memory
        relevant_games = df_meta["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_games)].copy()
        df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_games)].copy()

        # D. Preprocessing for Merge
        print("Preprocessing IDs and timestamps...")

        # Ensure IDs are strings for consistent merging (handles 'G' vs int issues)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)

        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)
        df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)

        # Calculate Frame for Helmets alignment
        # Step is 10Hz, Video is ~59.94Hz. Snap (step 0) is at 300 frames.
        # Formula: frame = 300 + step * 6
        df_meta["frame"] = (300 + df_meta["step"] * 6).astype(int)

        # Select columns to keep from tracking
        # Always keep keys: game_play, step, nfl_player_id
        tracking_cols_to_keep = [
            "game_play",
            "step",
            "nfl_player_id",
        ] + Config.TRACKING_BASE_COLS
        # Filter to only keep columns that actually exist in the raw file (some might be derived later)
        tracking_cols_to_keep = [
            c for c in tracking_cols_to_keep if c in df_tracking.columns
        ]
        df_tracking = df_tracking[tracking_cols_to_keep]

        # Select columns to keep from helmets
        # Keys: game_play, frame, nfl_player_id, view
        visual_cols_to_keep = [
            "game_play",
            "frame",
            "nfl_player_id",
            "view",
        ] + Config.VISUAL_BASE_COLS
        # Filter to only keep columns that actually exist in the raw file
        visual_cols_to_keep = [
            c for c in visual_cols_to_keep if c in df_helmets.columns
        ]
        df_helmets = df_helmets[visual_cols_to_keep]

        # E. Merging Tracking Data
        print("Merging tracking data...")

        # Merge Player 1 Tracking
        # Rename columns to avoid collisions and indicate player 1
        df_merged = pd.merge(
            df_meta,
            df_tracking.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )
        # Drop redundant key columns from merge
        df_merged = df_merged.drop(
            columns=["game_play_p1", "step_p1", "nfl_player_id_p1"]
        )

        # Merge Player 2 Tracking
        # Note: If Player 2 is 'G', these columns will be NaN, which is expected
        df_merged = pd.merge(
            df_merged,
            df_tracking.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )
        df_merged = df_merged.drop(
            columns=["game_play_p2", "step_p2", "nfl_player_id_p2"]
        )

        # F. Merging Helmet Data
        print("Merging helmet data...")

        # Split helmets by view to handle them as separate feature sets
        helmets_sideline = df_helmets[df_helmets["view"] == "Sideline"].drop(
            columns=["view"]
        )
        helmets_endzone = df_helmets[df_helmets["view"] == "Endzone"].drop(
            columns=["view"]
        )

        # Helper function to merge specific view and player
        def merge_view(df, helmets_subset, view_name):
            # helmets_subset has keys: game_play, frame, nfl_player_id

            # Merge for Player 1
            suffix_p1 = f"_{view_name}_p1"
            tmp_p1 = helmets_subset.add_suffix(suffix_p1)
            # Keys in tmp_p1: game_play_{suffix}, frame_{suffix}, nfl_player_id_{suffix}

            df = pd.merge(
                df,
                tmp_p1,
                left_on=["game_play", "frame", "nfl_player_id_1"],
                right_on=[
                    f"game_play{suffix_p1}",
                    f"frame{suffix_p1}",
                    f"nfl_player_id{suffix_p1}",
                ],
                how="left",
            )
            df = df.drop(
                columns=[
                    f"game_play{suffix_p1}",
                    f"frame{suffix_p1}",
                    f"nfl_player_id{suffix_p1}",
                ]
            )

            # Merge for Player 2
            suffix_p2 = f"_{view_name}_p2"
            tmp_p2 = helmets_subset.add_suffix(suffix_p2)

            df = pd.merge(
                df,
                tmp_p2,
                left_on=["game_play", "frame", "nfl_player_id_2"],
                right_on=[
                    f"game_play{suffix_p2}",
                    f"frame{suffix_p2}",
                    f"nfl_player_id{suffix_p2}",
                ],
                how="left",
            )
            df = df.drop(
                columns=[
                    f"game_play{suffix_p2}",
                    f"frame{suffix_p2}",
                    f"nfl_player_id{suffix_p2}",
                ]
            )

            return df

        # Apply merges for both views
        df_merged = merge_view(df_merged, helmets_sideline, "sideline")
        df_merged = merge_view(df_merged, helmets_endzone, "endzone")

        # 4. Save to Cache
        print(f"Saving merged {split} data to cache at {cache_path}...")
        save_to_cache(df_merged, cache_path)

        return df_merged
