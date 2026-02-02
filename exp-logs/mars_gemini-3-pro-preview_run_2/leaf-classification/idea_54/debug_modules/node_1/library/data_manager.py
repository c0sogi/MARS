import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CACHE_DIR,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
    INTERACTION_PAIRS,
    FLOAT_PRECISION,
)
from library.image_features import get_morphometric_features


class DataManager:
    def __init__(self, load_cached_data=True):
        """
        Initializes the DataManager.

        Args:
            load_cached_data (bool): Whether to attempt loading data from cache.
        """
        self.load_cached_data = load_cached_data
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.files = {
            "X_train": os.path.join(self.cache_dir, "X_train.npz"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.npz"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.npz"),
            "test_ids": os.path.join(self.cache_dir, "test_ids.npy"),
            "classes": os.path.join(self.cache_dir, "classes.npy"),
        }

    def load_data(self):
        """
        Loads the dataset, processing it from scratch or loading from cache.

        Returns:
            dict: A dictionary containing:
                - X_train (dict of arrays), y_train (array)
                - X_val (dict of arrays), y_val (array)
                - X_test (dict of arrays), test_ids (array)
                - classes (array of class names)
        """
        # 1. Try Loading from Cache
        if self.load_cached_data and self._check_cache_exists():
            print("Loading data from cache...")
            return self._load_from_cache()

        # 2. Process from Scratch
        print("Processing data from scratch...")

        # Load Metadata
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VAL_CSV)
        df_test = pd.read_csv(TEST_CSV)

        # Process Labels
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["species"]).astype(int)
        y_val = le.transform(df_val["species"]).astype(int)
        classes = le.classes_

        # Process Features (Views)
        X_train = self._extract_views(df_train, "train")
        X_val = self._extract_views(df_val, "val")
        X_test = self._extract_views(df_test, "test")

        test_ids = df_test["id"].values

        # 3. Save to Cache
        self._save_to_cache(X_train, y_train, X_val, y_val, X_test, test_ids, classes)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "test_ids": test_ids,
            "classes": classes,
        }

    def _extract_views(self, df, split_name):
        """
        Extracts all feature views (Global, Semantic, Morphometric, Interactions) for a dataframe.
        """
        views = {}

        # 1. Global View
        global_cols = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS
        views["global"] = df[global_cols].values.astype(FLOAT_PRECISION)

        # 2. Semantic Views (Individual Groups)
        views["margin"] = df[MARGIN_COLS].values.astype(FLOAT_PRECISION)
        views["shape"] = df[SHAPE_COLS].values.astype(FLOAT_PRECISION)
        views["texture"] = df[TEXTURE_COLS].values.astype(FLOAT_PRECISION)

        # 3. Morphometric View (Image Features)
        # Pass load_cached_data to the helper so it can use its own cache if available
        views["morphometric"] = get_morphometric_features(
            df, split_name, load_cached_data=self.load_cached_data
        )

        # 4. Interaction Views (Cross-Domain)
        # Defined in config: list of (name1, cols1, name2, cols2)
        for name1, _, name2, _ in INTERACTION_PAIRS:
            key = f"{name1}_{name2}"
            # Concatenate the arrays from the already extracted semantic views
            v1 = views[name1]
            v2 = views[name2]
            views[key] = np.hstack([v1, v2]).astype(FLOAT_PRECISION)

        return views

    def _check_cache_exists(self):
        """Checks if all required cache files exist."""
        return all(os.path.exists(path) for path in self.files.values())

    def _save_to_cache(self, X_train, y_train, X_val, y_val, X_test, test_ids, classes):
        """Saves processed data to disk using numpy formats."""
        print(f"Saving data to cache at {self.cache_dir}...")
        np.savez(self.files["X_train"], **X_train)
        np.save(self.files["y_train"], y_train)

        np.savez(self.files["X_val"], **X_val)
        np.save(self.files["y_val"], y_val)

        np.savez(self.files["X_test"], **X_test)
        np.save(self.files["test_ids"], test_ids)

        np.save(self.files["classes"], classes)

    def _load_from_cache(self):
        """Loads data from disk."""
        # np.load with allow_pickle=True is default for savez, but we use it for dict structure
        # We convert the NpzFile object to a standard dict for mutability and ease of use
        X_train = dict(np.load(self.files["X_train"]))
        y_train = np.load(self.files["y_train"])

        X_val = dict(np.load(self.files["X_val"]))
        y_val = np.load(self.files["y_val"])

        X_test = dict(np.load(self.files["X_test"]))
        test_ids = np.load(self.files["test_ids"])

        classes = np.load(self.files["classes"], allow_pickle=True)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "test_ids": test_ids,
            "classes": classes,
        }
