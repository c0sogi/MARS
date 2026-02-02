import os
import pandas as pd
from library.config import Config
from library.feature_engineering import FeatureEngine
from library.utils import seed_everything


class DataLoader:
    """
    Orchestrates the data loading pipeline for the VAAM-E strategy.

    This class serves as the primary interface for accessing processed datasets.
    It delegates the complex logic of merging, vector decomposition, spectral shock
    calculation, and relaxed quadratic gating to the FeatureEngine, while managing
    dataset selection and caching configuration.
    """

    def __init__(self):
        """
        Initialize the DataLoader.
        Sets random seeds to ensure deterministic behavior across the pipeline.
        """
        seed_everything(Config.SEED)
        self.feature_engine = FeatureEngine()

    def load_dataset(self, dataset_type, debug=False, load_cached_data=True):
        """
        Generic method to load, process, and return a dataset.

        This method resolves the appropriate cache path from the Config and
        invokes the FeatureEngine to perform the end-to-end data processing
        pipeline (Metadata Load -> Tracking Merge -> Feature Engineering -> Gating).

        Args:
            dataset_type (str): The type of dataset to load ('train', 'val', 'test').
            debug (bool): If True, samples the dataset (via metadata) for rapid debugging.
                          Controlled by Config.DEBUG_SAMPLE_SIZE.
            load_cached_data (bool): If True, attempts to load processed data from
                                     the cache defined in Config. If False or cache miss,
                                     recomputes features.

        Returns:
            pd.DataFrame: The fully processed, feature-engineered, and gated DataFrame.
        """
        # Validate dataset type and retrieve cache path from Config
        try:
            cache_path = Config.get_cache_path(dataset_type)
        except ValueError as e:
            raise ValueError(f"Invalid dataset_type '{dataset_type}': {e}")

        # Delegate to FeatureEngine.
        # The FeatureEngine.process_dataset method is decorated with @cache_processor,
        # which handles the check-load-compute-save cycle.
        # It also internally calls load_metadata (with debug sampling), _load_tracking_data,
        # and performs the merge and feature calculation.
        df = self.feature_engine.process_dataset(
            dataset_type=dataset_type,
            debug=debug,
            load_cached_data=load_cached_data,
            cache_path=cache_path,
        )

        return df

    def get_train_data(self, debug=False, load_cached_data=True):
        """
        Retrieves the training dataset.

        Args:
            debug (bool): Enable debug sampling.
            load_cached_data (bool): Enable caching.

        Returns:
            pd.DataFrame: Processed training data.
        """
        return self.load_dataset(
            "train", debug=debug, load_cached_data=load_cached_data
        )

    def get_val_data(self, debug=False, load_cached_data=True):
        """
        Retrieves the validation dataset.

        Args:
            debug (bool): Enable debug sampling.
            load_cached_data (bool): Enable caching.

        Returns:
            pd.DataFrame: Processed validation data.
        """
        return self.load_dataset("val", debug=debug, load_cached_data=load_cached_data)

    def get_test_data(self, debug=False, load_cached_data=True):
        """
        Retrieves the test dataset for inference.

        Args:
            debug (bool): Enable debug sampling.
            load_cached_data (bool): Enable caching.

        Returns:
            pd.DataFrame: Processed test data.
        """
        return self.load_dataset("test", debug=debug, load_cached_data=load_cached_data)
