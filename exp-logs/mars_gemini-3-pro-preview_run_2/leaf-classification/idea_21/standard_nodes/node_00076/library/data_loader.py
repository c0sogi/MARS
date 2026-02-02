import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from library.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    CACHE_TRAIN_IMG_FEATURES,
    CACHE_VAL_IMG_FEATURES,
    CACHE_TEST_IMG_FEATURES,
    ORIGINAL_FEATURE_COLS,
    MORPHOLOGICAL_COLS,
    ID_COL,
    TARGET_COL,
)
from library.feature_engineering import MorphologyExtractor


class LeafDataManager:
    """
    Orchestrates data loading, feature engineering, and preprocessing.
    Manages the creation of 'original', 'morphological', and 'combined' views
    for the Dynamic Ensemble Selection pipeline.
    """

    def __init__(self):
        self.extractor = MorphologyExtractor()

    def _load_metadata(self):
        """Loads the metadata CSVs."""
        if (
            not os.path.exists(TRAIN_FILE)
            or not os.path.exists(VAL_FILE)
            or not os.path.exists(TEST_FILE)
        ):
            raise FileNotFoundError(
                "Metadata files not found. Ensure metadata generation was successful."
            )

        df_train = pd.read_csv(TRAIN_FILE)
        df_val = pd.read_csv(VAL_FILE)
        df_test = pd.read_csv(TEST_FILE)
        return df_train, df_val, df_test

    def _process_features(self, df, cache_path, load_cached_data):
        """
        Extracts morphological features using the extractor, handling caching.
        Merges new features with the original dataframe.
        """
        # Extract morphological features (handles caching internally)
        morph_df = self.extractor.process_dataset(
            df, cache_path=cache_path, load_cached_data=load_cached_data
        )

        # Merge with original dataframe on ID
        # We use a left join to preserve the original dataframe structure
        merged_df = pd.merge(df, morph_df, on=ID_COL, how="left")

        # Fill any potential NaNs in the new features with 0.0 (though extractor should handle this)
        merged_df[MORPHOLOGICAL_COLS] = merged_df[MORPHOLOGICAL_COLS].fillna(0.0)

        return merged_df

    def _get_matrices(self, df, is_test=False):
        """
        Converts DataFrame columns to numpy arrays (float64) organized by view.
        Returns a dictionary of views and the target/id array.
        """
        # 1. Original View (Provided Histograms)
        X_original = df[ORIGINAL_FEATURE_COLS].values.astype(np.float64)

        # 2. Morphological View (Engineered Geometric Descriptors)
        X_morph = df[MORPHOLOGICAL_COLS].values.astype(np.float64)

        # 3. Combined View (Concatenation)
        X_combined = np.hstack([X_original, X_morph])

        X_dict = {
            "original": X_original,
            "morphological": X_morph,
            "combined": X_combined,
        }

        if is_test:
            ids = df[ID_COL].values
            return X_dict, ids
        else:
            y = df[TARGET_COL].values
            return X_dict, y

    def get_splits(self, load_cached_data=True):
        """
        Loads data, extracts features, and prepares Train/Val/Test splits.
        Used for the Selection Phase (Phase 1).

        Returns:
            X_train (dict), y_train, X_val (dict), y_val, X_test (dict), test_ids, classes
        """
        # 1. Load Metadata
        df_train, df_val, df_test = self._load_metadata()

        # 2. Feature Engineering (with Caching)
        df_train = self._process_features(
            df_train, CACHE_TRAIN_IMG_FEATURES, load_cached_data
        )
        df_val = self._process_features(
            df_val, CACHE_VAL_IMG_FEATURES, load_cached_data
        )
        df_test = self._process_features(
            df_test, CACHE_TEST_IMG_FEATURES, load_cached_data
        )

        # 3. Matrix Creation
        X_train, y_train = self._get_matrices(df_train, is_test=False)
        X_val, y_val = self._get_matrices(df_val, is_test=False)
        X_test, test_ids = self._get_matrices(df_test, is_test=True)

        classes = np.unique(y_train)

        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    def get_full_data(self, load_cached_data=True):
        """
        Loads data, extracts features, merges Train+Val, and prepares Full Train/Test splits.
        Used for the Final Retraining Phase (Phase 2).

        Returns:
            X_full (dict), y_full, X_test (dict), test_ids, classes
        """
        # 1. Load Metadata
        df_train, df_val, df_test = self._load_metadata()

        # 2. Feature Engineering (with Caching)
        df_train = self._process_features(
            df_train, CACHE_TRAIN_IMG_FEATURES, load_cached_data
        )
        df_val = self._process_features(
            df_val, CACHE_VAL_IMG_FEATURES, load_cached_data
        )
        df_test = self._process_features(
            df_test, CACHE_TEST_IMG_FEATURES, load_cached_data
        )

        # 3. Concatenate Train + Val
        df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

        # 4. Matrix Creation
        X_full, y_full = self._get_matrices(df_full, is_test=False)
        X_test, test_ids = self._get_matrices(df_test, is_test=True)

        classes = np.unique(y_full)

        return X_full, y_full, X_test, test_ids, classes
