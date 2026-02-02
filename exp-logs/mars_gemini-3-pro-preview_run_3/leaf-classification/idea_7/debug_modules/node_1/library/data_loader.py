import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.config import Config


class LeafDataManager:
    """
    Manages loading of tabular data and image paths for the leaf classification task.
    Handles caching of processed features and labels to disk using .npy format.
    """

    def __init__(self):
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Define local cache paths for items not explicitly in Config but necessary for the pipeline
        self.cache_train_ids = os.path.join(Config.CACHE_DIR, "train_ids.npy")
        self.cache_train_paths = os.path.join(Config.CACHE_DIR, "train_paths.npy")
        self.cache_test_paths = os.path.join(Config.CACHE_DIR, "test_paths.npy")

    def load_train_data(self, load_cached_data=True):
        """
        Loads the training data. Combines 'train' and 'val' metadata splits
        to maximize data usage for the Cross-Validation ensemble strategy.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X_tabular, y, file_paths, ids)
                X_tabular (np.ndarray): Tabular features (N, 192).
                y (np.ndarray): Encoded labels (N,).
                file_paths (np.ndarray): Relative image paths (N,).
                ids (np.ndarray): Image IDs (N,).
        """
        # Check if all necessary cache files exist
        cache_files = [
            Config.CACHE_TRAIN_TABULAR,
            Config.CACHE_TRAIN_LABELS,
            Config.CACHE_CLASSES,
            self.cache_train_ids,
            self.cache_train_paths,
        ]

        if load_cached_data and all(os.path.exists(f) for f in cache_files):
            print("Loading training data from cache...")
            X_tabular = np.load(Config.CACHE_TRAIN_TABULAR)
            y = np.load(Config.CACHE_TRAIN_LABELS)
            ids = np.load(self.cache_train_ids)
            # allow_pickle=True is required for string arrays (paths)
            file_paths = np.load(self.cache_train_paths, allow_pickle=True)
            return X_tabular, y, file_paths, ids

        print("Processing training data from metadata...")

        # Validate metadata existence
        if not os.path.exists(Config.TRAIN_METADATA) or not os.path.exists(
            Config.VAL_METADATA
        ):
            raise FileNotFoundError(
                "Metadata files not found. Ensure metadata generation script has run."
            )

        # Load and combine datasets
        df_train = pd.read_csv(Config.TRAIN_METADATA)
        df_val = pd.read_csv(Config.VAL_METADATA)
        df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

        # Extract Tabular Features based on Config prefixes
        feature_cols = []
        for prefix in Config.TABULAR_PREFIXES:
            # Find columns starting with the prefix (e.g., "margin", "shape")
            cols = [c for c in df_full.columns if c.startswith(prefix)]
            feature_cols.extend(cols)

        # Sort feature columns to ensure deterministic order between train and test
        feature_cols.sort()

        X_tabular = df_full[feature_cols].values.astype(np.float32)

        # Encode Labels
        le = LabelEncoder()
        y = le.fit_transform(df_full["species"])
        classes = le.classes_

        # Extract IDs and Paths
        ids = df_full["id"].values.astype(np.int64)
        file_paths = df_full["file_path"].values

        # Save to cache
        np.save(Config.CACHE_TRAIN_TABULAR, X_tabular)
        np.save(Config.CACHE_TRAIN_LABELS, y)
        np.save(Config.CACHE_CLASSES, classes)
        np.save(self.cache_train_ids, ids)
        np.save(self.cache_train_paths, file_paths)

        print(
            f"Training data processed and cached. Samples: {len(X_tabular)}, Classes: {len(classes)}"
        )

        return X_tabular, y, file_paths, ids

    def load_test_data(self, load_cached_data=True):
        """
        Loads the test data.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X_tabular, ids, file_paths)
                X_tabular (np.ndarray): Tabular features (N, 192).
                ids (np.ndarray): Image IDs (N,).
                file_paths (np.ndarray): Relative image paths (N,).
        """
        cache_files = [
            Config.CACHE_TEST_TABULAR,
            Config.CACHE_TEST_IDS,
            self.cache_test_paths,
        ]

        if load_cached_data and all(os.path.exists(f) for f in cache_files):
            print("Loading test data from cache...")
            X_tabular = np.load(Config.CACHE_TEST_TABULAR)
            ids = np.load(Config.CACHE_TEST_IDS)
            file_paths = np.load(self.cache_test_paths, allow_pickle=True)
            return X_tabular, ids, file_paths

        print("Processing test data from metadata...")

        if not os.path.exists(Config.TEST_METADATA):
            raise FileNotFoundError(
                f"Test metadata file not found at {Config.TEST_METADATA}"
            )

        df_test = pd.read_csv(Config.TEST_METADATA)

        # Extract Tabular Features
        feature_cols = []
        for prefix in Config.TABULAR_PREFIXES:
            cols = [c for c in df_test.columns if c.startswith(prefix)]
            feature_cols.extend(cols)

        # Sort to match training order
        feature_cols.sort()

        X_tabular = df_test[feature_cols].values.astype(np.float32)
        ids = df_test["id"].values.astype(np.int64)
        file_paths = df_test["file_path"].values

        # Save to cache
        np.save(Config.CACHE_TEST_TABULAR, X_tabular)
        np.save(Config.CACHE_TEST_IDS, ids)
        np.save(self.cache_test_paths, file_paths)

        print(f"Test data processed and cached. Samples: {len(X_tabular)}")

        return X_tabular, ids, file_paths

    def get_classes(self):
        """
        Returns the list of class names in the correct order (matching the encoded labels).
        If classes are not cached, it triggers loading of training data to generate them.

        Returns:
            np.ndarray: Array of class names (strings).
        """
        if os.path.exists(Config.CACHE_CLASSES):
            return np.load(Config.CACHE_CLASSES, allow_pickle=True)
        else:
            print(
                "Classes cache not found. Loading training data to generate classes..."
            )
            _, _, _, _ = self.load_train_data(load_cached_data=False)
            return np.load(Config.CACHE_CLASSES, allow_pickle=True)
