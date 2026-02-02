import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import CacheManager


class DataLoader:
    """
    Handles loading of metadata, tracking data, and helmet data.
    Implements caching for expensive merge operations.
    """

    def __init__(self, debug=Config.DEBUG):
        self.debug = debug
        self.cache_manager = CacheManager()

    def load_metadata(self, mode="train"):
        """
        Loads the metadata for the specified mode (train, validation, test).

        Args:
            mode (str): One of 'train', 'validation', 'test'.

        Returns:
            pd.DataFrame: The metadata dataframe.
        """
        if mode == "train":
            path = Config.TRAIN_META_PATH
        elif mode == "validation":
            path = Config.VAL_META_PATH
        elif mode == "test":
            path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)

        # In debug mode, sample a subset of plays
        if self.debug:
            unique_plays = df["game_play"].unique()
            # Sample 5% of plays or at least 5 plays
            n_sample = max(5, int(len(unique_plays) * 0.05))
            sampled_plays = np.random.choice(unique_plays, n_sample, replace=False)
            df = df[df["game_play"].isin(sampled_plays)].reset_index(drop=True)
            print(
                f"[DEBUG] Sampled {len(df)} rows from {mode} metadata ({len(sampled_plays)} plays)."
            )

        return df

    def load_tracking(self, mode="train", game_plays=None):
        """
        Loads player tracking data.

        Args:
            mode (str): 'train' (includes validation) or 'test'.
            game_plays (list, optional): List of game_play IDs to filter by.

        Returns:
            pd.DataFrame: The tracking dataframe.
        """
        # Determine file path based on mode
        # Note: Validation data comes from the training tracking file
        if mode in ["train", "validation"]:
            path = Config.TRAIN_TRACKING_PATH
        elif mode == "test":
            path = Config.TEST_TRACKING_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load data
        # We don't cache the raw load here because filtering by game_plays is fast
        # and the raw file is needed for different splits.
        df = pd.read_csv(path)

        # Standardize types
        df["nfl_player_id"] = df["nfl_player_id"].astype(str)

        # Filter if specific plays are requested
        if game_plays is not None:
            df = df[df["game_play"].isin(game_plays)].reset_index(drop=True)

        return df

    def load_helmets(self, mode="train", game_plays=None):
        """
        Loads helmet baseline predictions.

        Args:
            mode (str): 'train' (includes validation) or 'test'.
            game_plays (list, optional): List of game_play IDs to filter by.

        Returns:
            pd.DataFrame: The helmets dataframe.
        """
        if mode in ["train", "validation"]:
            path = Config.TRAIN_HELMETS_PATH
        elif mode == "test":
            path = Config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        df = pd.read_csv(path)

        if game_plays is not None:
            df = df[df["game_play"].isin(game_plays)].reset_index(drop=True)

        return df

    def merge_tracking_to_labels(self, labels_df, tracking_df, load_cached_data=True):
        """
        Merges tracking data onto the labels dataframe for both Player 1 and Player 2.

        Args:
            labels_df (pd.DataFrame): The labels/metadata dataframe.
            tracking_df (pd.DataFrame): The player tracking dataframe.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The merged dataframe with _p1 and _p2 tracking features.
        """
        # Generate a cache ID based on the input data characteristics
        # We use the length and the first/last values to create a signature
        # This is faster than hashing the whole dataframe
        labels_sig = f"{len(labels_df)}_{labels_df['game_play'].iloc[0] if not labels_df.empty else 'empty'}"
        tracking_sig = f"{len(tracking_df)}_{tracking_df['game_play'].iloc[0] if not tracking_df.empty else 'empty'}"

        config_dict = {
            "function": "merge_tracking_to_labels",
            "labels_sig": labels_sig,
            "tracking_sig": tracking_sig,
            "debug": self.debug,
        }

        cache_id = self.cache_manager.generate_cache_id(
            config_dict, prefix="merged_tracking"
        )

        # 1. Try to load from cache
        if load_cached_data:
            cached_df = self.cache_manager.load(cache_id, file_type="parquet")
            if cached_df is not None:
                return cached_df

        # 2. Perform Merge
        # Ensure ID types match for merging
        labels_df = labels_df.copy()
        tracking_df = tracking_df.copy()

        labels_df["nfl_player_id_1"] = labels_df["nfl_player_id_1"].astype(str)
        labels_df["nfl_player_id_2"] = labels_df["nfl_player_id_2"].astype(str)
        tracking_df["nfl_player_id"] = tracking_df["nfl_player_id"].astype(str)

        # Merge Player 1
        # Rename tracking columns to have _p1 suffix
        track_p1 = tracking_df.add_suffix("_p1")
        # The merge keys in tracking are game_play_p1, step_p1, nfl_player_id_p1
        # The merge keys in labels are game_play, step, nfl_player_id_1

        df_merged = pd.merge(
            labels_df,
            track_p1,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge Player 2
        # Rename tracking columns to have _p2 suffix
        track_p2 = tracking_df.add_suffix("_p2")

        df_merged = pd.merge(
            df_merged,
            track_p2,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Drop redundant merge columns to save space
        drop_cols = [
            "game_play_p1",
            "step_p1",
            "nfl_player_id_p1",
            "game_play_p2",
            "step_p2",
            "nfl_player_id_p2",
        ]
        df_merged.drop(
            columns=[c for c in drop_cols if c in df_merged.columns], inplace=True
        )

        # 3. Save to cache
        self.cache_manager.save(df_merged, cache_id, file_type="parquet")

        return df_merged

    def prepare_submission_skeleton(self):
        """
        Loads the test metadata which serves as the skeleton for submission.
        The metadata generation process has already parsed contact_id.

        Returns:
            pd.DataFrame: Test dataframe with parsed columns.
        """
        return self.load_metadata(mode="test")
