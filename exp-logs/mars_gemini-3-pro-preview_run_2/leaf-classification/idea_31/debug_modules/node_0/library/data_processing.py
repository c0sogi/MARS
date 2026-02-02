import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from library import config
from library import feature_engineering


class DataProcessor:
    """
    Handles data loading, preprocessing, view generation, and caching for the SCPGE strategy.
    Enforces float64 precision and applies PowerTransformer to all features.
    """

    def __init__(self, load_cached_data=True):
        self.load_cached_data = load_cached_data
        self.cache_dir = config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define mapping of data keys to cache filenames
        self.files = {
            "X_train_combined": "X_train_combined.npy",
            "X_train_global": "X_train_global.npy",
            "X_train_macro": "X_train_macro.npy",
            "y_train": "y_train.npy",
            "X_val_combined": "X_val_combined.npy",
            "X_val_global": "X_val_global.npy",
            "X_val_macro": "X_val_macro.npy",
            "y_val": "y_val.npy",
            "X_test_combined": "X_test_combined.npy",
            "X_test_global": "X_test_global.npy",
            "X_test_macro": "X_test_macro.npy",
            "test_ids": "test_ids.npy",
        }

    def get_data(self):
        """
        Orchestrates data retrieval.
        1. Checks cache.
        2. If missing/forced, loads raw data via feature_engineering.
        3. Applies PowerTransformer (Yeo-Johnson) and float64 casting.
        4. Slices data into Global (Micro) and Macro views.
        5. Caches and returns the processed data dictionary.
        """
        # Set seed for reproducibility (though processing is largely deterministic)
        np.random.seed(config.RANDOM_STATE)

        # 1. Check Cache
        all_cached = all(
            os.path.exists(os.path.join(self.cache_dir, fname))
            for fname in self.files.values()
        )

        if self.load_cached_data and all_cached:
            print(f"Loading processed data from cache: {self.cache_dir}")
            data = {}
            for key, fname in self.files.items():
                data[key] = np.load(
                    os.path.join(self.cache_dir, fname), allow_pickle=True
                )
            return data

        # 2. Process from Scratch
        print("Processing data from scratch...")

        # Load raw data (DataFrames)
        # Note: load_dataset handles the macro feature extraction internally
        (X_train_df, y_train, _), (X_val_df, y_val, _), (X_test_df, test_ids) = (
            feature_engineering.load_dataset(load_cached_data=self.load_cached_data)
        )

        # Identify feature columns
        all_cols = X_train_df.columns.tolist()

        # Micro features (Global View): provided 192 features (margin, shape, texture)
        micro_cols = [
            c
            for c in all_cols
            if any(c.startswith(p) for p in config.MICRO_FEATURE_PREFIXES)
        ]

        # Macro features (Macro View): extracted morphometrics
        macro_cols = [c for c in all_cols if c.startswith("macro_")]

        if not micro_cols:
            raise ValueError("No micro features found in dataset.")
        if not macro_cols:
            raise ValueError("No macro features found in dataset.")

        # Get indices for slicing later
        micro_indices = [X_train_df.columns.get_loc(c) for c in micro_cols]
        macro_indices = [X_train_df.columns.get_loc(c) for c in macro_cols]

        # Convert to float64 numpy arrays (Mandatory for SCPGE strategy)
        X_train_full = X_train_df.values.astype(config.FLOAT_PRECISION)
        X_val_full = X_val_df.values.astype(config.FLOAT_PRECISION)
        X_test_full = X_test_df.values.astype(config.FLOAT_PRECISION)

        # 3. Apply PowerTransformer (Yeo-Johnson)
        # We fit on the training set and transform everything.
        # We transform the 'Combined' set to handle all features at once.
        print("Applying PowerTransformer (Yeo-Johnson) to feature space...")
        pt = PowerTransformer(method="yeo-johnson", standardize=True)

        X_train_transformed = pt.fit_transform(X_train_full)
        X_val_transformed = pt.transform(X_val_full)
        X_test_transformed = pt.transform(X_test_full)

        # 4. Create Views and Construct Data Dictionary
        data = {
            # Combined Views
            "X_train_combined": X_train_transformed,
            "X_val_combined": X_val_transformed,
            "X_test_combined": X_test_transformed,
            # Global Views (Micro features only)
            "X_train_global": X_train_transformed[:, micro_indices],
            "X_val_global": X_val_transformed[:, micro_indices],
            "X_test_global": X_test_transformed[:, micro_indices],
            # Macro Views (Morphometrics only)
            "X_train_macro": X_train_transformed[:, macro_indices],
            "X_val_macro": X_val_transformed[:, macro_indices],
            "X_test_macro": X_test_transformed[:, macro_indices],
            # Targets and IDs
            "y_train": y_train,
            "y_val": y_val,
            "test_ids": test_ids,
        }

        # 5. Save to Cache
        print(f"Saving processed data to cache: {self.cache_dir}")
        for key, fname in self.files.items():
            np.save(os.path.join(self.cache_dir, fname), data[key])

        return data
