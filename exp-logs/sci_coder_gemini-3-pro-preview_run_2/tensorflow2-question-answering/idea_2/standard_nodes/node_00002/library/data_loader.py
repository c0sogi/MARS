import pandas as pd
from library.config import PathConfig
from library.feature_extractor import build_features_for_dataset


class NQDataReader:
    """
    Data loader for the Natural Questions dataset.
    Wraps the feature extraction pipeline to provide DataFrames for training, validation, and testing.
    Handles caching and split management via metadata files.
    """

    def get_training_samples(
        self, sample_size: int = None, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads training samples.
        Uses the training metadata to filter the source JSONL.
        Applies negative sampling (via is_train=True in feature extractor) to balance the dataset.

        Args:
            sample_size (int, optional): Number of documents to process. Useful for debugging.
            load_cached_data (bool): If True, attempts to load from Parquet cache.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors and labels for training.
        """
        return build_features_for_dataset(
            jsonl_path=PathConfig.TRAIN_JSONL,
            metadata_path=PathConfig.TRAIN_META,
            output_path=PathConfig.TRAIN_FEATURES_CACHE,
            is_train=True,
            load_cached_data=load_cached_data,
            sample_size=sample_size,
        )

    def get_validation_samples(
        self, sample_size: int = None, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads validation samples.
        Uses the validation metadata to filter the source JSONL.
        Processes ALL candidates (is_train=False) to allow for accurate ranking metric evaluation.

        Args:
            sample_size (int, optional): Number of documents to process.
            load_cached_data (bool): If True, attempts to load from Parquet cache.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors and labels for validation.
        """
        return build_features_for_dataset(
            jsonl_path=PathConfig.TRAIN_JSONL,
            metadata_path=PathConfig.VAL_META,
            output_path=PathConfig.VAL_FEATURES_CACHE,
            is_train=False,
            load_cached_data=load_cached_data,
            sample_size=sample_size,
        )

    def get_test_candidates(
        self, sample_size: int = None, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Loads test candidates for inference.
        Uses the test metadata and test JSONL file.
        Processes ALL candidates to generate features for ranking.

        Args:
            sample_size (int, optional): Number of documents to process.
            load_cached_data (bool): If True, attempts to load from Parquet cache.

        Returns:
            pd.DataFrame: DataFrame containing feature vectors for the test set.
        """
        return build_features_for_dataset(
            jsonl_path=PathConfig.TEST_JSONL,
            metadata_path=PathConfig.TEST_META,
            output_path=PathConfig.TEST_FEATURES_CACHE,
            is_train=False,
            load_cached_data=load_cached_data,
            sample_size=sample_size,
        )
