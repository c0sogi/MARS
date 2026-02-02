import os
import pandas as pd
from library import config
from library import utils
from library.feature_engineering import FeaturePipeline


class DataLoader:
    """
    Orchestrates the data loading and preparation process for the RAOP task.

    This class serves as the bridge between raw data/cached features and the
    model training pipeline. It delegates feature engineering and cleaning
    to the FeaturePipeline class while ensuring all necessary components
    (including test IDs for submission) are aggregated into a single data structure.
    """

    def __init__(self, load_cached_data: bool = True):
        """
        Initialize the DataLoader.

        Args:
            load_cached_data (bool): If True, attempts to load processed features
                                     from the cache directory defined in config.
                                     If False, forces recalculation of all features.
        """
        self.load_cached_data = load_cached_data
        self.logger = utils.get_logger(__name__)
        # Initialize the pipeline responsible for heavy feature engineering
        self.pipeline = FeaturePipeline(load_cached_data=load_cached_data)

    def load_data(self):
        """
        Executes the data loading workflow.

        1. Invokes FeaturePipeline to get cleaned, transformed, and vectorized features.
           This handles:
           - Metadata (scaled, retrieval columns removed)
           - Lexical (TF-IDF on concatenated text)
           - Behavioral (TF-IDF on subreddit history)
           - Semantic (Dense embeddings)
           - Target variables (y_train, y_val)

        2. Loads 'request_id' for the test set from the metadata parquet file.
           This is required for generating the submission file in the correct format.

        Returns:
            dict: A dictionary containing all data arrays:
                - 'y_train', 'y_val'
                - 'X_train_meta', 'X_val_meta', 'X_test_meta'
                - 'X_train_lexical', 'X_val_lexical', 'X_test_lexical'
                - 'X_train_behavioral', 'X_val_behavioral', 'X_test_behavioral'
                - 'X_train_semantic', 'X_val_semantic', 'X_test_semantic'
                - 'test_ids': numpy array of request_ids for the test set.
        """
        self.logger.info("Starting data loading sequence...")

        # Step 1: Retrieve processed features and targets via the pipeline
        # The pipeline handles the logic for caching and specific feature cleaning
        data = self.pipeline.execute()

        # Step 2: Retrieve Test IDs for Submission
        # The feature pipeline focuses on model inputs (X) and outputs (y).
        # We need to manually load the IDs to map predictions back to requests.
        self.logger.info(f"Loading test IDs from {config.TEST_PATH}...")

        if not os.path.exists(config.TEST_PATH):
            raise FileNotFoundError(f"Test metadata file missing at {config.TEST_PATH}")

        # Read only the ID column for efficiency
        test_df = pd.read_parquet(config.TEST_PATH, columns=[config.ID_COL])
        data["test_ids"] = test_df[config.ID_COL].values

        self.logger.info(
            f"Data loading complete. Loaded {len(data['test_ids'])} test samples."
        )

        return data
