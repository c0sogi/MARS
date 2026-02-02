import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger
from library.features import FeatureExtractor


class VolcanoDataset:
    """
    Manages data loading and preparation for the Volcano Eruption Prediction Task.
    Utilizes FeatureExtractor to transform raw sensor data into feature matrices
    using the 'Hybrid-Transform Orthogonal Decomposition' strategy.
    """

    def __init__(self):
        self.logger = setup_logger("VolcanoDataset")
        self.extractor = FeatureExtractor()

    def _load_and_process(
        self,
        meta_path: str,
        dataset_name: str,
        load_cached_data: bool,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Internal helper to load metadata and process features via FeatureExtractor.

        Args:
            meta_path: Path to the metadata CSV.
            dataset_name: Base name for the dataset (e.g., 'train').
            load_cached_data: Whether to try loading from parquet cache.
            limit: Number of rows to process (for debugging).

        Returns:
            pd.DataFrame: The processed feature dataframe.
        """
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        meta_df = pd.read_csv(meta_path)

        # Handle limiting/debugging
        # If a limit is applied, we must change the dataset_name to avoid
        # overwriting the full dataset cache with a partial one.
        effective_name = dataset_name
        if limit is not None:
            self.logger.info(f"Limiting {dataset_name} dataset to {limit} samples.")
            meta_df = meta_df.head(limit)
            effective_name = f"{dataset_name}_debug_{limit}"

        # Delegate to FeatureExtractor which handles parallel processing,
        # memory optimization, and caching logic.
        features_df = self.extractor.process_data(
            meta_df=meta_df,
            dataset_name=effective_name,
            load_cached_data=load_cached_data,
        )

        return features_df

    def get_train_data(self, load_cached_data: bool = True, limit: int = None):
        """
        Loads and processes the training dataset.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector (time_to_eruption).
        """
        self.logger.info("Retrieving Training Data...")
        df = self._load_and_process(
            Config.TRAIN_META_PATH, "train", load_cached_data, limit
        )

        if "time_to_eruption" not in df.columns:
            raise ValueError("Training data missing 'time_to_eruption' column.")

        y = df["time_to_eruption"]

        # Drop non-feature columns
        drop_cols = ["segment_id", "time_to_eruption"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return X, y

    def get_val_data(self, load_cached_data: bool = True, limit: int = None):
        """
        Loads and processes the validation dataset.

        Returns:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector (time_to_eruption).
        """
        self.logger.info("Retrieving Validation Data...")
        df = self._load_and_process(
            Config.VAL_META_PATH, "val", load_cached_data, limit
        )

        if "time_to_eruption" not in df.columns:
            raise ValueError("Validation data missing 'time_to_eruption' column.")

        y = df["time_to_eruption"]

        drop_cols = ["segment_id", "time_to_eruption"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return X, y

    def get_test_data(self, load_cached_data: bool = True, limit: int = None):
        """
        Loads and processes the test dataset.

        Returns:
            X (pd.DataFrame): Feature matrix.
            ids (pd.Series): Segment IDs corresponding to the features.
        """
        self.logger.info("Retrieving Test Data...")
        df = self._load_and_process(
            Config.TEST_META_PATH, "test", load_cached_data, limit
        )

        if "segment_id" not in df.columns:
            raise ValueError("Test data missing 'segment_id' column.")

        ids = df["segment_id"]

        # Drop non-feature columns (target shouldn't be there, but segment_id is)
        drop_cols = ["segment_id", "time_to_eruption"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return X, ids
