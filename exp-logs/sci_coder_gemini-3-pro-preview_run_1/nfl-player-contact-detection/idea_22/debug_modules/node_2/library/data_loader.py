import pandas as pd
import numpy as np
from library.config import Config
from library.features import FeatureEngineer


class DataLoader:
    """
    Handles data ingestion, merging, and preprocessing by orchestrating the
    FeatureEngineer library. Implements the pipeline for loading metadata,
    merging with tracking data, and applying relaxed quadratic gating via
    the vector-decomposed physics engine.
    """

    def __init__(self):
        """
        Initializes the DataLoader.
        Configuration is derived from library.config.Config.
        """
        pass

    def load_train_data(self, load_cached_data=True, sample_size=None):
        """
        Loads the training dataset.

        This process involves:
        1. Loading train_metadata.csv and train_player_tracking.csv.
        2. Merging data and computing vector-decomposed kinematic features.
        3. Applying Relaxed Quadratic Reachability Gating to filter unlikely contacts.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.
            sample_size (int, optional): If provided, returns a random sample of the dataset
                                         for debugging purposes.

        Returns:
            pd.DataFrame: The processed and gated training dataset.
        """
        engineer = FeatureEngineer(
            metadata_path=Config.TRAIN_METADATA_PATH,
            tracking_path=Config.TRAIN_TRACKING_PATH,
        )

        # generate_features handles loading, preprocessing, merging, physics, gating, and caching
        df = engineer.generate_features(load_cached_data=load_cached_data)

        if sample_size is not None and len(df) > sample_size:
            print(
                f"Sampling {sample_size} rows from training data (Total: {len(df)})..."
            )
            df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(
                drop=True
            )

        return df

    def load_val_data(self, load_cached_data=True, sample_size=None):
        """
        Loads the validation dataset.

        Uses val_metadata.csv but shares the train_player_tracking.csv source.
        Applies the same feature engineering and gating logic as training data.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.
            sample_size (int, optional): If provided, returns a random sample.

        Returns:
            pd.DataFrame: The processed and gated validation dataset.
        """
        engineer = FeatureEngineer(
            metadata_path=Config.VAL_METADATA_PATH,
            tracking_path=Config.TRAIN_TRACKING_PATH,
        )

        df = engineer.generate_features(load_cached_data=load_cached_data)

        if sample_size is not None and len(df) > sample_size:
            print(
                f"Sampling {sample_size} rows from validation data (Total: {len(df)})..."
            )
            df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(
                drop=True
            )

        return df

    def load_test_data(self, load_cached_data=True):
        """
        Loads the test dataset for inference.

        Uses test_metadata.csv and test_player_tracking.csv.
        Note: Relaxed Quadratic Gating is typically skipped or handled differently
        for test data in the FeatureEngineer to ensure all submission rows are present,
        or the model must handle missing rows by predicting 0.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: The processed test dataset with features.
        """
        engineer = FeatureEngineer(
            metadata_path=Config.TEST_METADATA_PATH,
            tracking_path=Config.TEST_TRACKING_PATH,
        )

        # generate_features automatically detects test mode via path inspection
        # and skips gating to preserve all contact_ids for submission.
        df = engineer.generate_features(load_cached_data=load_cached_data)

        return df
