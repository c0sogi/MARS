import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import reduce_mem_usage, save_parquet, load_parquet


class DataFactory:
    """
    Factory class responsible for loading, merging, and preprocessing the raw
    NFL Contact Detection datasets. Implements caching and specific handling
    for Player-Player vs Player-Ground interactions.
    """

    @staticmethod
    def load_metadata(path):
        """
        Loads the metadata CSV file containing contact labels and video paths.

        Args:
            path (str): Path to the metadata CSV.

        Returns:
            pd.DataFrame: Loaded and memory-optimized dataframe.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)

        # Ensure consistent types for merge keys
        df["game_play"] = df["game_play"].astype(str)
        df["step"] = df["step"].astype(int)
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(int)
        # nfl_player_id_2 can be int or 'G', keep as string/object initially
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)

        df = reduce_mem_usage(df, verbose=False)
        return df

    @staticmethod
    def load_tracking(path):
        """
        Loads the player tracking CSV file.

        Args:
            path (str): Path to the tracking CSV.

        Returns:
            pd.DataFrame: Loaded and memory-optimized dataframe.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found: {path}")

        df = pd.read_csv(path)

        # Ensure consistent types for merge keys
        df["game_play"] = df["game_play"].astype(str)
        df["step"] = df["step"].astype(int)
        df["nfl_player_id"] = df["nfl_player_id"].astype(int)

        # Rename 'distance' (distance traveled) to avoid conflict with pairwise distance
        if "distance" in df.columns:
            df = df.rename(columns={"distance": "distance_traveled"})

        df = reduce_mem_usage(df, verbose=False)
        return df

    @staticmethod
    def _merge_tracking(df_meta, df_track):
        """
        Merges tracking data onto metadata for both players involved in the contact_id.
        Handles the distinction between Player-Player and Player-Ground interactions.

        Args:
            df_meta (pd.DataFrame): Metadata dataframe.
            df_track (pd.DataFrame): Tracking dataframe.

        Returns:
            pd.DataFrame: Merged dataframe with _p1 and _p2 tracking features.
        """
        # Identify tracking columns to merge (excluding keys)
        track_cols = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]

        # ---------------------------------------------------------
        # 1. Split into Player-Player (PP) and Player-Ground (PG)
        # ---------------------------------------------------------
        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_pg = df_meta[mask_ground].copy()
        df_pp = df_meta[~mask_ground].copy()

        # ---------------------------------------------------------
        # 2. Process Player-Player (PP) Interactions
        # ---------------------------------------------------------
        if not df_pp.empty:
            # Convert ID 2 to int for merging
            df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

            # Merge Player 1 Tracking
            df_pp = pd.merge(
                df_pp,
                df_track,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            # Rename P1 columns
            rename_p1 = {c: f"{c}_p1" for c in track_cols}
            df_pp = df_pp.rename(columns=rename_p1)
            df_pp = df_pp.drop(columns=["nfl_player_id"])

            # Merge Player 2 Tracking
            df_pp = pd.merge(
                df_pp,
                df_track,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            # Rename P2 columns
            rename_p2 = {c: f"{c}_p2" for c in track_cols}
            df_pp = df_pp.rename(columns=rename_p2)
            df_pp = df_pp.drop(columns=["nfl_player_id"])

            # Calculate Euclidean Distance for PP
            # Ensure coordinates are float
            coords = [
                "x_position_p1",
                "y_position_p1",
                "x_position_p2",
                "y_position_p2",
            ]
            for col in coords:
                df_pp[col] = df_pp[col].astype(float)

            df_pp["distance"] = np.sqrt(
                (df_pp["x_position_p1"] - df_pp["x_position_p2"]) ** 2
                + (df_pp["y_position_p1"] - df_pp["y_position_p2"]) ** 2
            )

        # ---------------------------------------------------------
        # 3. Process Player-Ground (PG) Interactions
        # ---------------------------------------------------------
        if not df_pg.empty:
            # Merge Player 1 Tracking
            df_pg = pd.merge(
                df_pg,
                df_track,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            rename_p1 = {c: f"{c}_p1" for c in track_cols}
            df_pg = df_pg.rename(columns=rename_p1)
            df_pg = df_pg.drop(columns=["nfl_player_id"])

            # Handle Player 2 (Ground)
            # Fill P2 tracking columns with 0.0 to maintain schema consistency
            for col in track_cols:
                df_pg[f"{col}_p2"] = 0.0

            # Set Sentinel Distance for Ground (Critical for Tree Split)
            df_pg["distance"] = Config.GROUND_DISTANCE_SENTINEL

        # ---------------------------------------------------------
        # 4. Recombine and Finalize
        # ---------------------------------------------------------
        df_final = pd.concat([df_pp, df_pg], axis=0, ignore_index=True)

        # Sort by contact_id to ensure deterministic ordering
        if "contact_id" in df_final.columns:
            df_final = df_final.sort_values("contact_id").reset_index(drop=True)

        df_final = reduce_mem_usage(df_final, verbose=True)
        return df_final

    @staticmethod
    def prepare_base_data(split="train", load_cached_data=True, sample_size=None):
        """
        Orchestrates the loading and merging of data for a specific split.
        Implements caching to speed up subsequent runs.

        Args:
            split (str): One of 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from cache first.
            sample_size (int, optional): If provided, returns a subset of the data.

        Returns:
            pd.DataFrame: The prepared dataset.
        """
        # Determine paths and cache filename
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            track_path = Config.TRAIN_TRACKING_PATH
            cache_name = "merged_train.parquet"
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
            # Validation uses the train tracking file (subset of plays)
            track_path = Config.TRAIN_TRACKING_PATH
            cache_name = "merged_val.parquet"
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
            track_path = Config.TEST_TRACKING_PATH
            cache_name = "merged_test.parquet"
        else:
            raise ValueError(f"Invalid split: {split}")

        # Attempt to load from cache
        if load_cached_data:
            df_cached = load_parquet(cache_name)
            if df_cached is not None:
                print(f"[{split}] Loaded base data from cache: {cache_name}")
                if sample_size:
                    return df_cached.head(sample_size)
                return df_cached

        print(f"[{split}] Processing base data from scratch...")

        # Load raw files
        df_meta = DataFactory.load_metadata(meta_path)
        df_track = DataFactory.load_tracking(track_path)

        # Merge
        df_merged = DataFactory._merge_tracking(df_meta, df_track)

        # Save to cache
        save_parquet(df_merged, cache_name)
        print(f"[{split}] Saved processed data to cache: {cache_name}")

        # Clean up memory
        del df_meta, df_track
        gc.collect()

        if sample_size:
            return df_merged.head(sample_size)

        return df_merged
