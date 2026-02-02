import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import reduce_mem_usage


class DataLoader:
    """
    Handles data ingestion, formatting, and caching for the NFL Contact Detection task.
    """

    def __init__(self):
        self.config = Config

    def load_metadata(self):
        """
        Loads the pre-split train, validation, and test metadata files.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        print("Loading metadata...")

        # Load Train
        if os.path.exists(self.config.TRAIN_META_PATH):
            train_df = pd.read_csv(self.config.TRAIN_META_PATH)
            # Ensure player IDs are strings to match tracking/helmet data
            train_df["nfl_player_id_1"] = train_df["nfl_player_id_1"].astype(str)
            train_df["nfl_player_id_2"] = train_df["nfl_player_id_2"].astype(str)
        else:
            raise FileNotFoundError(
                f"Train metadata not found at {self.config.TRAIN_META_PATH}"
            )

        # Load Validation
        if os.path.exists(self.config.VAL_META_PATH):
            val_df = pd.read_csv(self.config.VAL_META_PATH)
            val_df["nfl_player_id_1"] = val_df["nfl_player_id_1"].astype(str)
            val_df["nfl_player_id_2"] = val_df["nfl_player_id_2"].astype(str)
        else:
            raise FileNotFoundError(
                f"Validation metadata not found at {self.config.VAL_META_PATH}"
            )

        # Load Test
        if os.path.exists(self.config.TEST_META_PATH):
            test_df = pd.read_csv(self.config.TEST_META_PATH)
            test_df["nfl_player_id_1"] = test_df["nfl_player_id_1"].astype(str)
            test_df["nfl_player_id_2"] = test_df["nfl_player_id_2"].astype(str)
        else:
            raise FileNotFoundError(
                f"Test metadata not found at {self.config.TEST_META_PATH}"
            )

        print(
            f"Metadata loaded. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
        )
        return train_df, val_df, test_df

    def load_tracking_data(self, load_cached_data=True):
        """
        Loads player tracking data. Uses parquet caching to speed up subsequent loads.

        Args:
            load_cached_data (bool): If True, attempts to load from cached parquet files.

        Returns:
            tuple: (train_tracking_df, test_tracking_df)
        """
        cache_dir = self.config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        train_cache_path = os.path.join(cache_dir, "train_tracking.parquet")
        test_cache_path = os.path.join(cache_dir, "test_tracking.parquet")

        # --- Train Tracking ---
        if load_cached_data and os.path.exists(train_cache_path):
            print(f"Loading cached train tracking data from {train_cache_path}...")
            train_tracking = pd.read_parquet(train_cache_path)
        else:
            print(
                f"Loading raw train tracking data from {self.config.TRAIN_TRACKING_PATH}..."
            )
            train_tracking = pd.read_csv(self.config.TRAIN_TRACKING_PATH)

            # Type conversion
            if "nfl_player_id" in train_tracking.columns:
                train_tracking["nfl_player_id"] = train_tracking[
                    "nfl_player_id"
                ].astype(str)

            train_tracking = reduce_mem_usage(train_tracking)
            train_tracking.to_parquet(train_cache_path, index=False)

        # --- Test Tracking ---
        if load_cached_data and os.path.exists(test_cache_path):
            print(f"Loading cached test tracking data from {test_cache_path}...")
            test_tracking = pd.read_parquet(test_cache_path)
        else:
            print(
                f"Loading raw test tracking data from {self.config.TEST_TRACKING_PATH}..."
            )
            test_tracking = pd.read_csv(self.config.TEST_TRACKING_PATH)

            # Type conversion
            if "nfl_player_id" in test_tracking.columns:
                test_tracking["nfl_player_id"] = test_tracking["nfl_player_id"].astype(
                    str
                )

            test_tracking = reduce_mem_usage(test_tracking)
            test_tracking.to_parquet(test_cache_path, index=False)

        return train_tracking, test_tracking

    def load_helmet_data(self, load_cached_data=True):
        """
        Loads baseline helmet data. Uses parquet caching.

        Args:
            load_cached_data (bool): If True, attempts to load from cached parquet files.

        Returns:
            tuple: (train_helmets_df, test_helmets_df)
        """
        cache_dir = self.config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        train_cache_path = os.path.join(cache_dir, "train_helmets.parquet")
        test_cache_path = os.path.join(cache_dir, "test_helmets.parquet")

        # --- Train Helmets ---
        if load_cached_data and os.path.exists(train_cache_path):
            print(f"Loading cached train helmet data from {train_cache_path}...")
            train_helmets = pd.read_parquet(train_cache_path)
        else:
            print(
                f"Loading raw train helmet data from {self.config.TRAIN_HELMETS_PATH}..."
            )
            train_helmets = pd.read_csv(self.config.TRAIN_HELMETS_PATH)

            # Type conversion
            if "nfl_player_id" in train_helmets.columns:
                train_helmets["nfl_player_id"] = train_helmets["nfl_player_id"].astype(
                    str
                )

            train_helmets = reduce_mem_usage(train_helmets)
            train_helmets.to_parquet(train_cache_path, index=False)

        # --- Test Helmets ---
        if load_cached_data and os.path.exists(test_cache_path):
            print(f"Loading cached test helmet data from {test_cache_path}...")
            test_helmets = pd.read_parquet(test_cache_path)
        else:
            print(
                f"Loading raw test helmet data from {self.config.TEST_HELMETS_PATH}..."
            )
            test_helmets = pd.read_csv(self.config.TEST_HELMETS_PATH)

            # Type conversion
            if "nfl_player_id" in test_helmets.columns:
                test_helmets["nfl_player_id"] = test_helmets["nfl_player_id"].astype(
                    str
                )

            test_helmets = reduce_mem_usage(test_helmets)
            test_helmets.to_parquet(test_cache_path, index=False)

        return train_helmets, test_helmets

    @staticmethod
    def parse_contact_id(df):
        """
        Parses the 'contact_id' column into constituent parts: game_play, step, nfl_player_id_1, nfl_player_id_2.
        Useful for processing raw submission files or creating new splits.

        Args:
            df (pd.DataFrame): DataFrame containing a 'contact_id' column.

        Returns:
            pd.DataFrame: DataFrame with added columns.
        """
        if "contact_id" not in df.columns:
            raise ValueError("DataFrame must contain 'contact_id' column.")

        # contact_id format: game_key_play_id_step_player1_player2
        # Example: 58168_003392_0_38590_43854

        split_data = df["contact_id"].str.split("_", expand=True)

        # Check if split resulted in expected number of columns (5)
        if split_data.shape[1] != 5:
            # Handle potential edge cases or malformed IDs if necessary,
            # but for this dataset structure is consistent.
            pass

        df["game_play"] = split_data[0] + "_" + split_data[1]
        df["step"] = split_data[2].astype(int)
        df["nfl_player_id_1"] = split_data[3].astype(str)
        df["nfl_player_id_2"] = split_data[4].astype(str)

        return df
