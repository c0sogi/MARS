import pandas as pd
import numpy as np
from library.features import FeatureFactory
from library.utils import seed_everything
from library.config import SEED


class NFLDataLoader:
    """
    Data Loader class that orchestrates the loading and preprocessing of data
    for the NFL Player Contact Detection task.

    This class serves as an interface to the FeatureFactory, which handles
    the heavy lifting of data ingestion, feature engineering, caching,
    stream splitting, and undersampling.
    """

    def __init__(self):
        self.feature_factory = FeatureFactory()

    def load_raw_data(self):
        """
        Conceptually handles loading of raw CSVs.
        Implementation is delegated to FeatureFactory._load_tracking internally.
        """
        pass

    def merge_tracking_and_labels(self):
        """
        Conceptually handles the merging of tracking data with contact labels.
        Implementation is delegated to FeatureFactory internally.
        """
        pass

    def prepare_streams(self, split="train", load_cached_data=True):
        """
        Prepares Stream A (Player-Player) and Stream B (Player-Ground) data.

        This method utilizes the FeatureFactory to:
        1. Check for cached processed data using a configuration hash.
        2. If not cached, load raw data, engineer features, and split streams.
        3. Apply random undersampling to the training set.
        4. Save/Load the result to/from disk.

        Args:
            split (str): The dataset split to load ('train', 'validation', or 'test').
            load_cached_data (bool): If True, attempts to load pre-processed data
                                     from the working directory cache.

        Returns:
            dict: A dictionary containing the processed data for both streams:
                {
                    "stream_a": {"X": pd.DataFrame, "y": np.ndarray, "ids": np.ndarray},
                    "stream_b": {"X": pd.DataFrame, "y": np.ndarray, "ids": np.ndarray}
                }
        """
        # Ensure reproducibility for any stochastic operations (like undersampling)
        seed_everything(SEED)

        # The FeatureFactory.process_data method encapsulates the requirements:
        # - Caching via utils.get_config_hash
        # - Loading metadata and tracking data
        # - Splitting into Stream A and Stream B
        # - Feature engineering (flattened windows)
        # - Undersampling (for train split)
        return self.feature_factory.process_data(
            split=split, load_cached_data=load_cached_data
        )
