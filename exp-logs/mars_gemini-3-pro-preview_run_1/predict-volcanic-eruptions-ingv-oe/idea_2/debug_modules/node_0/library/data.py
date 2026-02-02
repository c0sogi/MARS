import pandas as pd
from library.config import TRAIN_METADATA_PATH, VAL_METADATA_PATH, TEST_METADATA_PATH
from library.features import FeatureManager


class VolcanoDataLoader:
    """
    Data loader for the Volcano Eruption Prediction task.
    Manages loading of metadata and retrieval of features via FeatureManager.
    """

    def __init__(self):
        self.feature_manager = FeatureManager()

    def get_train_data(self, load_cached_data=True, sample_size=None):
        """
        Loads training data (features and targets).

        Args:
            load_cached_data (bool): Whether to load features from cache if available.
            sample_size (int, optional): If set, returns a subset of the data for debugging.

        Returns:
            tuple: (X_train, y_train) where X_train is a DataFrame and y_train is a Series.
        """
        # Load metadata
        df_meta = pd.read_csv(TRAIN_METADATA_PATH)

        # Retrieve features (computes or loads from cache)
        # We pass the full metadata to FeatureManager to ensure the cache file
        # is generated for the complete dataset, preserving cache integrity.
        X, y = self.feature_manager.get_train_data(
            df_meta, load_cached_data=load_cached_data
        )

        # Apply sampling if requested
        if sample_size is not None and sample_size < len(X):
            X = X.iloc[:sample_size]
            y = y.iloc[:sample_size]

        return X, y

    def get_val_data(self, load_cached_data=True, sample_size=None):
        """
        Loads validation data (features and targets).

        Args:
            load_cached_data (bool): Whether to load features from cache if available.
            sample_size (int, optional): If set, returns a subset of the data.

        Returns:
            tuple: (X_val, y_val)
        """
        df_meta = pd.read_csv(VAL_METADATA_PATH)

        X, y = self.feature_manager.get_val_data(
            df_meta, load_cached_data=load_cached_data
        )

        if sample_size is not None and sample_size < len(X):
            X = X.iloc[:sample_size]
            y = y.iloc[:sample_size]

        return X, y

    def get_test_data(self, load_cached_data=True, sample_size=None):
        """
        Loads test data (features and segment_ids).

        Args:
            load_cached_data (bool): Whether to load features from cache if available.
            sample_size (int, optional): If set, returns a subset of the data.

        Returns:
            tuple: (X_test, segment_ids) where segment_ids is a Series of IDs needed for submission.
        """
        df_meta = pd.read_csv(TEST_METADATA_PATH)

        # get_test_data returns X and dummy y (zeros)
        X, _ = self.feature_manager.get_test_data(
            df_meta, load_cached_data=load_cached_data
        )

        # Extract segment_ids corresponding to the rows.
        # FeatureManager uses a left join on the metadata to merge features,
        # so the row order of X corresponds exactly to the row order of df_meta.
        segment_ids = df_meta["segment_id"]

        if sample_size is not None and sample_size < len(X):
            X = X.iloc[:sample_size]
            segment_ids = segment_ids.iloc[:sample_size]

        return X, segment_ids
