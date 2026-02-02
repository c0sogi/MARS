import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.utils import seed_everything


class LeafDataManager:
    def __init__(self, metadata_dir="./metadata", cache_dir="./working/idea_1"):
        """
        Initializes the data manager with paths to metadata and cache.
        """
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        self.train_path = os.path.join(metadata_dir, "train.csv")
        self.val_path = os.path.join(metadata_dir, "val.csv")
        self.test_path = os.path.join(metadata_dir, "test.csv")

        # Placeholders for data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.test_ids = None
        self.classes = None

        seed_everything()

    def _get_feature_columns(self, df):
        """
        Identifies feature columns based on prefixes (margin, shape, texture).
        """
        return [c for c in df.columns if c.startswith(("margin", "shape", "texture"))]

    def process_data(self, load_cached_data=True):
        """
        Loads, preprocesses, and caches the data.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.
        """
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        cache_paths = {
            "X_train": os.path.join(self.cache_dir, "X_train.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.npy"),
            "test_ids": os.path.join(self.cache_dir, "test_ids.npy"),
            "classes": os.path.join(self.cache_dir, "classes.npy"),
        }

        # Check if all cache files exist
        cache_exists = all(os.path.exists(p) for p in cache_paths.values())

        # Logic Flow 1: Try to load from cache
        if load_cached_data and cache_exists:
            print("Loading data from cache...")
            try:
                self.X_train = np.load(cache_paths["X_train"])
                self.y_train = np.load(cache_paths["y_train"])
                self.X_val = np.load(cache_paths["X_val"])
                self.y_val = np.load(cache_paths["y_val"])
                self.X_test = np.load(cache_paths["X_test"])
                self.test_ids = np.load(cache_paths["test_ids"])
                # allow_pickle=True is required for string arrays (classes)
                self.classes = np.load(cache_paths["classes"], allow_pickle=True)
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Logic Flow 2: Compute/Process from scratch
        print("Processing data from scratch...")

        # Load Metadata CSVs
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train metadata not found at {self.train_path}")

        df_train = pd.read_csv(self.train_path)
        df_val = pd.read_csv(self.val_path)
        df_test = pd.read_csv(self.test_path)

        # Extract Features
        feature_cols = self._get_feature_columns(df_train)

        X_train_raw = df_train[feature_cols].values
        y_train_raw = df_train["species"].values

        X_val_raw = df_val[feature_cols].values
        y_val_raw = df_val["species"].values

        X_test_raw = df_test[feature_cols].values
        self.test_ids = df_test["id"].values

        # Label Encoding
        le = LabelEncoder()
        # Fit on training labels. Metadata check ensured val classes are subset of train classes.
        self.y_train = le.fit_transform(y_train_raw)
        self.y_val = le.transform(y_val_raw)
        self.classes = le.classes_

        # Feature Scaling (StandardScaler)
        # Fit on training data only to prevent data leakage
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train_raw)
        self.X_val = scaler.transform(X_val_raw)
        self.X_test = scaler.transform(X_test_raw)

        # Save to cache
        print("Saving data to cache...")
        np.save(cache_paths["X_train"], self.X_train)
        np.save(cache_paths["y_train"], self.y_train)
        np.save(cache_paths["X_val"], self.X_val)
        np.save(cache_paths["y_val"], self.y_val)
        np.save(cache_paths["X_test"], self.X_test)
        np.save(cache_paths["test_ids"], self.test_ids)
        np.save(cache_paths["classes"], self.classes)

    def get_train_data(self, max_samples=None):
        """
        Returns the training data (X, y).

        Args:
            max_samples (int, optional): If provided, returns a subset of the data.
        """
        if self.X_train is None:
            raise ValueError("Data not processed. Call process_data() first.")

        if max_samples is not None and max_samples < len(self.X_train):
            return self.X_train[:max_samples], self.y_train[:max_samples]
        return self.X_train, self.y_train

    def get_val_data(self, max_samples=None):
        """
        Returns the validation data (X, y).

        Args:
            max_samples (int, optional): If provided, returns a subset of the data.
        """
        if self.X_val is None:
            raise ValueError("Data not processed. Call process_data() first.")

        if max_samples is not None and max_samples < len(self.X_val):
            return self.X_val[:max_samples], self.y_val[:max_samples]
        return self.X_val, self.y_val

    def get_test_data(self):
        """
        Returns the test data (X, ids).
        """
        if self.X_test is None:
            raise ValueError("Data not processed. Call process_data() first.")
        return self.X_test, self.test_ids

    def get_classes(self):
        """
        Returns the list of class names corresponding to the encoded labels.
        """
        if self.classes is None:
            raise ValueError("Data not processed. Call process_data() first.")
        return self.classes
