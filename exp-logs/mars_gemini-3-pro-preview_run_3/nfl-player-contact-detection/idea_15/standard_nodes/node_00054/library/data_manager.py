import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class DataLoader:
    """
    Handles data loading, global preprocessing, and caching.
    Specifically implements the Nearest-Neighbor Context generation for Stream B.
    """

    def __init__(self):
        self.logger = setup_logger(name="DataLoader")

    def load_metadata(self):
        """
        Loads the train, validation, and test metadata CSVs generated in the previous step.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        self.logger.info("Loading metadata...")
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        self.logger.info(
            f"Metadata loaded. Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )
        return df_train, df_val, df_test

    def load_helmets(self, dataset_type="train"):
        """
        Loads the baseline helmet detection data.

        Args:
            dataset_type (str): 'train' or 'test'.

        Returns:
            pd.DataFrame: Helmet data.
        """
        path = (
            Config.TRAIN_HELMETS_PATH
            if dataset_type == "train"
            else Config.TEST_HELMETS_PATH
        )
        self.logger.info(f"Loading helmets from {path}...")
        df = pd.read_csv(path)
        return df

    def get_processed_tracking(self, dataset_type="train", load_cached_data=True):
        """
        Loads tracking data.
        """
        cache_filename = f"processed_tracking_{dataset_type}.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Loading processed tracking data from cache: {cache_path}"
            )
            try:
                df_tracking = pd.read_parquet(cache_path)
                return df_tracking
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        self.logger.info(f"Loading raw tracking data for {dataset_type}...")

        # Load raw data
        raw_path = (
            Config.TRAIN_TRACKING_PATH
            if dataset_type == "train"
            else Config.TEST_TRACKING_PATH
        )
        df_tracking = pd.read_csv(raw_path, usecols=Config.RAW_TRACKING_COLS)

        # 3. Save to cache
        self.logger.info(f"Saving processed tracking data to cache: {cache_path}")
        df_tracking.to_parquet(cache_path, index=False)

        return df_tracking
