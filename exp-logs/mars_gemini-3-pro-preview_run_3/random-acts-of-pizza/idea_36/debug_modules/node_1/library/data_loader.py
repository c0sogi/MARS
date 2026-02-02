import pandas as pd
from library.config import Config
from library.utils import get_logger
from library.feature_engineering import process_data, TextProcessor

logger = get_logger(__name__)


class DataLoader:
    """
    Handles data ingestion and initial preprocessing.
    Delegates complex feature engineering and caching to the library.feature_engineering module.
    """

    def load_dataset(
        self, load_cached_data: bool = True, debug_sample_size: int = None
    ):
        """
        Loads the dataset (train, val, test) with all features engineered.

        This function utilizes the feature_engineering.process_data function which implements
        the strict caching logic (Parquet/Numpy), text vectorization, and metadata extraction.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from cache.
                                     If False or cache is missing/incomplete, re-processes data
                                     and saves to cache.
            debug_sample_size (int, optional): If set, limits the data size for debugging purposes.

        Returns:
            dict: A dictionary containing:
                - 'train': {'metadata': DataFrame, 'tfidf': sparse, 'embeddings': npy, 'y': npy}
                - 'val': {'metadata': DataFrame, 'tfidf': sparse, 'embeddings': npy, 'y': npy}
                - 'test': {'metadata': DataFrame, 'tfidf': sparse, 'embeddings': npy, 'ids': npy}
                - 'CommunityProfiler': The class for Bayesian community profiling.
        """
        logger.info("DataLoader: invoking feature engineering pipeline.")
        # process_data handles loading from Config.TRAIN_DATA_PATH (Parquet) and caching logic
        return process_data(
            load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
        )

    def preprocess_raw_data(self, df: pd.DataFrame, text_cols: list):
        """
        Applies TextProcessor cleaning rules (e.g., stripping edits) and prepares text.

        Args:
            df (pd.DataFrame): The dataframe containing raw text columns.
            text_cols (list): The list of columns to process and concatenate.

        Returns:
            pd.Series: The processed and concatenated text series.
        """
        # Utilizes the TextProcessor from the provided library to ensure consistency
        # with the training pipeline's text handling.
        return TextProcessor.process(df, text_cols)
