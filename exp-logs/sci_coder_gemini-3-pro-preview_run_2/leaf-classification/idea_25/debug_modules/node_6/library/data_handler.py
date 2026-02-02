import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_TRAIN_FEATURES,
    CACHE_VAL_FEATURES,
    CACHE_TEST_FEATURES,
    WORKING_DIR,
    USE_FLOAT64,
)
from library.image_processing import extract_robust_morphometrics


class DataManager:
    """
    Manages data loading, feature merging, view construction, and preprocessing.
    Implements caching for final processed arrays to optimize runtime.
    """

    def __init__(self):
        self.le = LabelEncoder()

    def load_data(self, load_cached_data=True):
        """
        Main entry point to load and process data.

        Args:
            load_cached_data (bool): If True, attempts to load processed numpy arrays from disk.

        Returns:
            tuple: (X_train_views, y_train, X_val_views, y_val, X_test_views, test_ids, classes)
                   X_*_views are dictionaries with keys ['Global', 'Morph', 'Combined'].
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(WORKING_DIR, "processed_data.npz")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return self._load_from_cache(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing data...")

        # 2. Load Metadata
        df_train = pd.read_csv(TRAIN_METADATA_PATH)
        df_val = pd.read_csv(VAL_METADATA_PATH)
        df_test = pd.read_csv(TEST_METADATA_PATH)

        # Fit LabelEncoder on full training data before any subsampling (Cite debug_lesson_8)
        self.le.fit(df_train["species"])
        classes = self.le.classes_

        # 3. Extract/Load Monte-Carlo Features
        # The image processor handles its own parquet-based caching
        mc_train = extract_robust_morphometrics(
            df_train, CACHE_TRAIN_FEATURES, load_cached_data
        )
        mc_val = extract_robust_morphometrics(
            df_val, CACHE_VAL_FEATURES, load_cached_data
        )
        mc_test = extract_robust_morphometrics(
            df_test, CACHE_TEST_FEATURES, load_cached_data
        )

        # 4. Merge Features
        # Merge on 'id' to combine provided features with generated morphometrics
        # Use inner join to ensure we only keep rows that were processed (important for DEBUG_MODE)
        df_train = df_train.merge(mc_train, on="id", how="inner")
        df_val = df_val.merge(mc_val, on="id", how="inner")
        df_test = df_test.merge(mc_test, on="id", how="inner")

        # 5. Prepare Targets and IDs
        y_train = self.le.transform(df_train["species"])
        y_val = self.le.transform(df_val["species"])
        test_ids = df_test["id"].values

        # 6. Define Feature Views
        # Global: Provided features (margin, shape, texture)
        global_cols = [
            c
            for c in df_train.columns
            if any(x in c for x in ["margin", "shape", "texture"])
        ]

        # Morph: Generated Monte-Carlo features (exclude id)
        morph_cols = [c for c in mc_train.columns if c != "id"]

        # Combined: Concatenation of both
        combined_cols = global_cols + morph_cols

        views_config = {
            "Global": global_cols,
            "Morph": morph_cols,
            "Combined": combined_cols,
        }

        X_train_views = {}
        X_val_views = {}
        X_test_views = {}

        # 7. Process Each View
        for view_name, cols in views_config.items():
            # Extract raw data
            raw_train = df_train[cols].values
            raw_val = df_val[cols].values
            raw_test = df_test[cols].values

            # Cast to float64 for precision
            if USE_FLOAT64:
                raw_train = raw_train.astype(np.float64)
                raw_val = raw_val.astype(np.float64)
                raw_test = raw_test.astype(np.float64)

            # Apply PowerTransformer (Yeo-Johnson)
            # Fit on Train, Transform on Val and Test
            pt = PowerTransformer(method="yeo-johnson")
            X_train_views[view_name] = pt.fit_transform(raw_train)
            X_val_views[view_name] = pt.transform(raw_val)
            X_test_views[view_name] = pt.transform(raw_test)

        # 8. Save to Cache
        save_dict = {
            "y_train": y_train,
            "y_val": y_val,
            "test_ids": test_ids,
            "classes": classes,
        }
        # Flatten dictionary for storage
        for view_name in views_config:
            save_dict[f"X_train_{view_name}"] = X_train_views[view_name]
            save_dict[f"X_val_{view_name}"] = X_val_views[view_name]
            save_dict[f"X_test_{view_name}"] = X_test_views[view_name]

        np.savez(cache_path, **save_dict)

        return (
            X_train_views,
            y_train,
            X_val_views,
            y_val,
            X_test_views,
            test_ids,
            classes,
        )

    def _load_from_cache(self, cache_path):
        """Helper to reconstruct dictionaries from flat npz file."""
        print(f"Loading processed data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)

        y_train = data["y_train"]
        y_val = data["y_val"]
        test_ids = data["test_ids"]
        classes = data["classes"]

        X_train_views = {}
        X_val_views = {}
        X_test_views = {}

        for view in ["Global", "Morph", "Combined"]:
            X_train_views[view] = data[f"X_train_{view}"]
            X_val_views[view] = data[f"X_val_{view}"]
            X_test_views[view] = data[f"X_test_{view}"]

        return (
            X_train_views,
            y_train,
            X_val_views,
            y_val,
            X_test_views,
            test_ids,
            classes,
        )
