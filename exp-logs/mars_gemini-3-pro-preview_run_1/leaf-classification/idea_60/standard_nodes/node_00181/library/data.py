import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from library import config, utils, features


class LeafDataManager:
    """
    Manages data loading, feature extraction, and the high-precision preprocessing pipeline.
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        # Pipeline components
        self.sanitizer = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
        self.transformer = (
            PowerTransformer(method="yeo-johnson", standardize=False)
            if config.USE_YEO_JOHNSON
            else None
        )
        self.scaler = StandardScaler() if config.USE_STANDARD_SCALING else None

    def _get_feature_columns(self, df):
        """
        Identifies feature columns by excluding metadata columns.
        Returns sorted list of column names for deterministic ordering.
        """
        exclude = {config.ID_COL, config.TARGET_COL, config.FILE_PATH_COL}
        feature_cols = [c for c in df.columns if c not in exclude]
        return sorted(feature_cols)

    def prepare_data(self, load_cached_data=True, debug_limit=None):
        """
        Main orchestration method.
        1. Checks for cached preprocessed numpy arrays.
        2. If not found or forced reload:
           - Loads raw data (via library.features).
           - Extracts features and targets.
           - Runs the Inductive Preprocessing Pipeline (Sanitize -> Transform -> Scale).
           - Caches the results.
        3. Returns the processed arrays.
        """

        # Define cache paths for the final processed arrays
        cache_paths = {
            "X_train": config.CACHE_X_TRAIN,
            "y_train": config.CACHE_Y_TRAIN,
            "X_val": config.CACHE_X_VAL,
            "y_val": config.CACHE_Y_VAL,
            "X_test": config.CACHE_X_TEST,
            "test_ids": config.CACHE_TEST_IDS,
            "classes": config.CACHE_CLASSES,
        }

        # 1. Check if all cache files exist
        all_cached = all(os.path.exists(path) for path in cache_paths.values())

        if load_cached_data and all_cached:
            utils.Logger.info("Loading preprocessed data from cache...")
            X_train = utils.load_cache_npy(cache_paths["X_train"])
            y_train = utils.load_cache_npy(cache_paths["y_train"])
            X_val = utils.load_cache_npy(cache_paths["X_val"])
            y_val = utils.load_cache_npy(cache_paths["y_val"])
            X_test = utils.load_cache_npy(cache_paths["X_test"])
            test_ids = utils.load_cache_npy(cache_paths["test_ids"])
            classes = utils.load_cache_npy(cache_paths["classes"])

            return X_train, y_train, X_val, y_val, X_test, test_ids, classes

        # 2. Compute from scratch
        utils.Logger.info("Starting data preparation pipeline from scratch...")

        # A. Load Raw Data (with features extracted)
        # library.features handles the extraction and its own parquet caching
        df_train = features.process_dataset(
            "train", load_cached_data=load_cached_data, debug_limit=debug_limit
        )
        df_val = features.process_dataset(
            "val", load_cached_data=load_cached_data, debug_limit=debug_limit
        )
        df_test = features.process_dataset(
            "test", load_cached_data=load_cached_data, debug_limit=debug_limit
        )

        # B. Extract Columns
        # Ensure deterministic column order
        feature_cols = self._get_feature_columns(df_train)
        utils.Logger.info(f"Identified {len(feature_cols)} feature columns.")

        # C. Prepare Raw Arrays (float64)
        X_train_raw = df_train[feature_cols].values.astype(config.FLOAT_PRECISION)
        X_val_raw = df_val[feature_cols].values.astype(config.FLOAT_PRECISION)
        X_test_raw = df_test[feature_cols].values.astype(config.FLOAT_PRECISION)

        y_train_raw = df_train[config.TARGET_COL].values
        y_val_raw = df_val[config.TARGET_COL].values
        test_ids = df_test[config.ID_COL].values

        # D. Encode Targets
        utils.Logger.info("Encoding targets...")
        self.label_encoder.fit(y_train_raw)

        # Filter validation data to remove unseen labels (Cite debug_lesson_1)
        known_classes = set(self.label_encoder.classes_)
        val_mask = np.array([label in known_classes for label in y_val_raw])

        if not np.all(val_mask):
            n_dropped = np.sum(~val_mask)
            utils.Logger.info(
                f"Debug Mode: Dropping {n_dropped} validation samples with unseen labels."
            )
            X_val_raw = X_val_raw[val_mask]
            y_val_raw = y_val_raw[val_mask]

        y_train = self.label_encoder.transform(y_train_raw)
        y_val = self.label_encoder.transform(y_val_raw)
        classes = self.label_encoder.classes_

        # E. Inductive Preprocessing Pipeline
        # Fit on TRAIN, Apply to TRAIN, VAL, TEST

        # Step 1: Sanitization (Variance Threshold)
        utils.Logger.info("Pipeline Step 1: Sanitization (VarianceThreshold)...")
        # Fit
        self.sanitizer.fit(X_train_raw)
        # Transform
        X_train_san = self.sanitizer.transform(X_train_raw)
        X_val_san = self.sanitizer.transform(X_val_raw)
        X_test_san = self.sanitizer.transform(X_test_raw)

        utils.Logger.metric("Original Features", X_train_raw.shape[1])
        utils.Logger.metric("Sanitized Features", X_train_san.shape[1])

        # Step 2: Transformation (Yeo-Johnson)
        current_X_train = X_train_san
        current_X_val = X_val_san
        current_X_test = X_test_san

        if self.transformer:
            utils.Logger.info("Pipeline Step 2: Transformation (Yeo-Johnson)...")
            # Fit
            self.transformer.fit(current_X_train)
            # Transform
            current_X_train = self.transformer.transform(current_X_train)
            current_X_val = self.transformer.transform(current_X_val)
            current_X_test = self.transformer.transform(current_X_test)

        # Step 3: Scaling (StandardScaler)
        if self.scaler:
            utils.Logger.info("Pipeline Step 3: Scaling (StandardScaler)...")
            # Fit
            self.scaler.fit(current_X_train)
            # Transform
            current_X_train = self.scaler.transform(current_X_train)
            current_X_val = self.scaler.transform(current_X_val)
            current_X_test = self.scaler.transform(current_X_test)

        # Final Arrays
        X_train = current_X_train
        X_val = current_X_val
        X_test = current_X_test

        # F. Cache Results
        utils.save_cache_npy(cache_paths["X_train"], X_train)
        utils.save_cache_npy(cache_paths["y_train"], y_train)
        utils.save_cache_npy(cache_paths["X_val"], X_val)
        utils.save_cache_npy(cache_paths["y_val"], y_val)
        utils.save_cache_npy(cache_paths["X_test"], X_test)
        utils.save_cache_npy(cache_paths["test_ids"], test_ids)
        utils.save_cache_npy(cache_paths["classes"], classes)

        return X_train, y_train, X_val, y_val, X_test, test_ids, classes
