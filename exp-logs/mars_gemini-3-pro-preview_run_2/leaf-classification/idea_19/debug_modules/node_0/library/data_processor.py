import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import Config


class DataProcessor:
    def __init__(self):
        """
        Initializes the DataProcessor with feature column definitions.
        """
        # Define the 192 feature columns based on dataset description
        self.margin_cols = [f"margin_{i}" for i in range(1, 65)]
        self.shape_cols = [f"shape_{i}" for i in range(1, 65)]
        self.texture_cols = [f"texture_{i}" for i in range(1, 65)]
        self.feature_cols = self.margin_cols + self.shape_cols + self.texture_cols

    def _extract_features(self, df):
        """
        Extracts the 192 feature columns as a numpy array.

        Args:
            df (pd.DataFrame): The dataframe containing feature columns.

        Returns:
            np.ndarray: A float32 array of shape (n_samples, 192).
        """
        return df[self.feature_cols].values.astype(np.float32)

    def load_and_process_data(self, load_cached_data=True):
        """
        Loads data from metadata CSVs, processes features and targets,
        and handles caching using numpy files.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from cache.

        Returns:
            dict: A dictionary containing:
                - X_train, y_train: Training features and labels
                - X_val, y_val: Validation features and labels
                - X_test: Test features
                - test_ids: IDs for the test set
                - classes: Array of original class names
        """
        # Define cache file paths
        cache_files = {
            "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
            "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
            "X_val": os.path.join(Config.CACHE_DIR, "X_val.npy"),
            "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
            "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
            "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
            "classes": os.path.join(Config.CACHE_DIR, "classes.npy"),
        }

        # Check if cache exists and should be loaded
        if load_cached_data:
            all_exist = all(os.path.exists(path) for path in cache_files.values())
            if all_exist:
                print("Loading data from cache...")
                data = {
                    k: np.load(v, allow_pickle=True) for k, v in cache_files.items()
                }
                return data

        print("Processing data from scratch...")

        # Load raw CSVs from metadata
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            raise FileNotFoundError(
                f"Training data not found at {Config.TRAIN_DATA_PATH}"
            )

        df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
        df_val = pd.read_csv(Config.VAL_DATA_PATH)
        df_test = pd.read_csv(Config.TEST_DATA_PATH)

        # Handle Debug Mode
        if Config.DEBUG:
            print(f"DEBUG mode: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
            df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
            df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

        # Extract Features
        X_train = self._extract_features(df_train)
        X_val = self._extract_features(df_val)
        X_test = self._extract_features(df_test)

        # Extract and Encode Targets
        le = LabelEncoder()
        # Fit on training species.
        y_train = le.fit_transform(df_train[Config.TARGET_COL])
        y_val = le.transform(df_val[Config.TARGET_COL])
        classes = le.classes_

        # Extract Test IDs
        test_ids = df_test[Config.ID_COL].values

        # Save to Cache
        Config.ensure_directories()
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["X_val"], X_val)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["test_ids"], test_ids)
        np.save(cache_files["classes"], classes)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "test_ids": test_ids,
            "classes": classes,
        }
