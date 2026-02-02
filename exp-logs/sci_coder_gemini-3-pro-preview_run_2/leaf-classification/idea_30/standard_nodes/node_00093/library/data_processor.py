import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from library.config import Config
from library.image_features import generate_macro_features


class LeafDataManager:
    """
    Manages data loading, feature generation, view construction, and preprocessing
    for the Precision-Covariance Multi-Resolution Ensemble.
    """

    def __init__(self):
        self.cache_file = os.path.join(Config.WORKING_DIR, "processed_data.npz")
        self.data = {}

    def load_and_process_data(self, load_cached_data=True):
        """
        Main entry point to load and process all data.

        Args:
            load_cached_data (bool): If True, attempts to load pre-processed matrices from disk.

        Returns:
            dict: A dictionary containing 'train', 'val', 'test' sub-dictionaries with X views, y, and ids.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_file):
            print(f"Loading processed data from cache: {self.cache_file}")
            try:
                self._load_from_cache()
                return self.data
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing from scratch...")

        # 2. Compute from scratch
        print("Starting data processing pipeline...")

        # Load Metadata
        df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
        df_val = pd.read_csv(Config.VAL_DATA_PATH)
        df_test = pd.read_csv(Config.TEST_DATA_PATH)

        # Debugging: Slice datasets if requested
        if Config.DEBUG_SAMPLE_SIZE is not None:
            print(f"DEBUG: Slicing datasets to {Config.DEBUG_SAMPLE_SIZE} samples.")
            df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
            df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]
            df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Generate Macro Features (Hu Moments + Geometric Scalars)
        # The library function handles its own caching of raw features via Parquet
        print("Generating/Loading Macro-Resolution features...")
        macro_train = generate_macro_features(
            df_train, Config.CACHE_TRAIN_MACRO, load_cached_data
        )
        macro_val = generate_macro_features(
            df_val, Config.CACHE_VAL_MACRO, load_cached_data
        )
        macro_test = generate_macro_features(
            df_test, Config.CACHE_TEST_MACRO, load_cached_data
        )

        # Merge Macro features back to main dataframes on 'id' to ensure alignment
        df_train = df_train.merge(macro_train, on="id", how="left")
        df_val = df_val.merge(macro_val, on="id", how="left")
        df_test = df_test.merge(macro_test, on="id", how="left")

        # Extract Feature Views
        print("Constructing feature views (Global, Macro, Combined)...")
        views_train = self._extract_views(df_train, is_test=False)
        views_val = self._extract_views(df_val, is_test=False)
        views_test = self._extract_views(df_test, is_test=True)

        # Extract Targets and IDs
        y_train_raw = df_train["species"].values
        y_val_raw = df_val["species"].values

        ids_train = df_train["id"].values
        ids_val = df_val["id"].values
        ids_test = df_test["id"].values

        # Label Encoding
        print("Encoding labels...")
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        # Handle potential unseen labels in val (unlikely given stratified split, but safe practice)
        # We map val labels based on train classes.
        # If a class is in val but not train, this would error.
        # Given the dataset constraints, we assume train covers all classes.
        y_val = le.transform(y_val_raw)

        classes = le.classes_

        # Preprocessing (PowerTransformer + Float64)
        print("Applying rigorous preprocessing (PowerTransformer + float64)...")
        processed_train, processed_val, processed_test = self._apply_preprocessing(
            views_train, views_val, views_test
        )

        # Structure Data
        self.data = {
            "train": {
                "X_global": processed_train["global"],
                "X_macro": processed_train["macro"],
                "X_combined": processed_train["combined"],
                "y": y_train,
                "ids": ids_train,
            },
            "val": {
                "X_global": processed_val["global"],
                "X_macro": processed_val["macro"],
                "X_combined": processed_val["combined"],
                "y": y_val,
                "ids": ids_val,
            },
            "test": {
                "X_global": processed_test["global"],
                "X_macro": processed_test["macro"],
                "X_combined": processed_test["combined"],
                "ids": ids_test,
            },
            "classes": classes,
        }

        # Save to Cache
        self._save_to_cache()

        print("Data processing complete.")
        return self.data

    def _extract_views(self, df, is_test=False):
        """
        Extracts Global, Macro, and Combined views from the dataframe.
        """
        # Identify columns
        # Global: Provided features (margin, shape, texture)
        feature_cols = [
            c
            for c in df.columns
            if c.startswith(Config.PREFIX_MARGIN)
            or c.startswith(Config.PREFIX_SHAPE)
            or c.startswith(Config.PREFIX_TEXTURE)
        ]

        # Macro: Extracted features (hu_, aspect_ratio, etc.)
        # We identify them by excluding metadata and global features
        exclude_cols = ["id", "species", "image_path"] + feature_cols
        macro_cols = [c for c in df.columns if c not in exclude_cols]

        # Sort columns to ensure consistent order
        feature_cols.sort()
        macro_cols.sort()

        X_global = df[feature_cols].values.astype(Config.FLOAT_TYPE)
        X_macro = df[macro_cols].values.astype(Config.FLOAT_TYPE)

        # Combined: Concatenate
        X_combined = np.hstack([X_global, X_macro])

        return {"global": X_global, "macro": X_macro, "combined": X_combined}

    def _apply_preprocessing(self, train_views, val_views, test_views):
        """
        Fits PowerTransformer on Train views and transforms Train, Val, and Test.
        Enforces float64 precision.
        """
        processed_train = {}
        processed_val = {}
        processed_test = {}

        for view_name in ["global", "macro", "combined"]:
            # Initialize Transformer
            pt = PowerTransformer(method="yeo-johnson", standardize=True)

            # Fit on Train
            # Note: PowerTransformer outputs float64 by default
            X_train_trans = pt.fit_transform(train_views[view_name])

            # Transform Val and Test
            X_val_trans = pt.transform(val_views[view_name])
            X_test_trans = pt.transform(test_views[view_name])

            # Explicitly cast to configured float type (float64) to be absolutely sure
            processed_train[view_name] = X_train_trans.astype(Config.FLOAT_TYPE)
            processed_val[view_name] = X_val_trans.astype(Config.FLOAT_TYPE)
            processed_test[view_name] = X_test_trans.astype(Config.FLOAT_TYPE)

        return processed_train, processed_val, processed_test

    def _save_to_cache(self):
        """
        Saves the processed data dictionary to an npz file.
        """
        print(f"Saving processed data to {self.cache_file}...")
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        np.savez(
            self.cache_file,
            # Train
            train_X_global=self.data["train"]["X_global"],
            train_X_macro=self.data["train"]["X_macro"],
            train_X_combined=self.data["train"]["X_combined"],
            train_y=self.data["train"]["y"],
            train_ids=self.data["train"]["ids"],
            # Val
            val_X_global=self.data["val"]["X_global"],
            val_X_macro=self.data["val"]["X_macro"],
            val_X_combined=self.data["val"]["X_combined"],
            val_y=self.data["val"]["y"],
            val_ids=self.data["val"]["ids"],
            # Test
            test_X_global=self.data["test"]["X_global"],
            test_X_macro=self.data["test"]["X_macro"],
            test_X_combined=self.data["test"]["X_combined"],
            test_ids=self.data["test"]["ids"],
            # Meta
            classes=self.data["classes"],
        )

    def _load_from_cache(self):
        """
        Loads data from npz file and reconstructs the dictionary structure.
        """
        loaded = np.load(self.cache_file, allow_pickle=True)

        self.data = {
            "train": {
                "X_global": loaded["train_X_global"],
                "X_macro": loaded["train_X_macro"],
                "X_combined": loaded["train_X_combined"],
                "y": loaded["train_y"],
                "ids": loaded["train_ids"],
            },
            "val": {
                "X_global": loaded["val_X_global"],
                "X_macro": loaded["val_X_macro"],
                "X_combined": loaded["val_X_combined"],
                "y": loaded["val_y"],
                "ids": loaded["val_ids"],
            },
            "test": {
                "X_global": loaded["test_X_global"],
                "X_macro": loaded["test_X_macro"],
                "X_combined": loaded["test_X_combined"],
                "ids": loaded["test_ids"],
            },
            "classes": loaded["classes"],
        }
