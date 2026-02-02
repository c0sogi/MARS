import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from library.config import (
    CACHE_DIR,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    TABULAR_FEATURE_GROUPS,
    GEOMETRIC_FEATURES,
    FLOAT_PRECISION,
    PREPROCESSING_PARAMS,
    SEED,
)
from library.feature_extraction import GeometricFeatureExtractor
from library.utils import set_seed


class SanitizedGroupPreprocessor:
    """
    Handles data loading, geometric feature extraction, group splitting,
    sanitization (variance threshold), and high-precision transformation.
    """

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.tabular_groups = TABULAR_FEATURE_GROUPS
        self.geometric_features = GEOMETRIC_FEATURES
        self.float_dtype = FLOAT_PRECISION
        self.params = PREPROCESSING_PARAMS

    def _get_cache_paths(self):
        """Defines paths for all cached numpy arrays."""
        paths = {}
        splits = ["train", "val", "test"]
        groups = list(self.tabular_groups.keys()) + ["geometry"]

        for split in splits:
            for group in groups:
                filename = f"X_{split}_{group}.npy"
                paths[f"X_{split}_{group}"] = os.path.join(self.cache_dir, filename)

        paths["y_train"] = os.path.join(self.cache_dir, "y_train.npy")
        paths["y_val"] = os.path.join(self.cache_dir, "y_val.npy")
        paths["test_ids"] = os.path.join(self.cache_dir, "test_ids.npy")
        paths["classes"] = os.path.join(self.cache_dir, "classes.npy")

        return paths

    def _load_and_merge_geometry(self, meta_path, split_name):
        """Loads metadata and merges extracted geometric features."""
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)

        # Initialize extractor with the dataframe
        extractor = GeometricFeatureExtractor(df)

        # Extract features (uses its own caching mechanism)
        # We append the split name to the cache file to avoid collisions if any
        geo_df = extractor.extract_features(
            load_cached_data=True, cache_name=f"geometric_features_{split_name}.parquet"
        )

        # Merge geometric features into the main dataframe
        # Both should have 'id' column
        df_merged = df.merge(geo_df, on="id", how="left")

        return df_merged

    def _extract_group_features(self, df, group_name):
        """Extracts specific feature columns for a group and converts to float64."""
        if group_name == "geometry":
            cols = self.geometric_features
        else:
            # Tabular groups are identified by prefix
            prefix = self.tabular_groups[group_name]
            # Filter columns that start with the prefix (e.g., 'margin_')
            cols = [c for c in df.columns if str(c).startswith(prefix)]

        # Sort columns to ensure consistent feature order
        cols = sorted(cols)

        # Return as float64 numpy array
        return df[cols].values.astype(self.float_dtype)

    def _create_pipeline(self):
        """Creates the preprocessing pipeline based on config."""
        steps = []

        # 1. Variance Threshold (Sanitization)
        if self.params.get("variance_threshold") is not None:
            vt = VarianceThreshold(threshold=self.params["variance_threshold"])
            steps.append(("vt", vt))

        # 2. Yeo-Johnson Power Transformation
        if self.params.get("yeo_johnson"):
            # standardize=False is crucial as per instructions
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            steps.append(("pt", pt))

        # 3. Standard Scaler
        if self.params.get("standardize"):
            ss = StandardScaler()
            steps.append(("ss", ss))

        return Pipeline(steps)

    def process_and_cache(self, load_cached_data=True):
        """
        Main method to process data.
        Checks cache, otherwise loads raw, computes features, transforms, and caches.
        """
        set_seed(SEED)
        os.makedirs(self.cache_dir, exist_ok=True)
        paths = self._get_cache_paths()

        # Check if all cache files exist
        all_cached = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_cached:
            try:
                # Load from cache
                # Use allow_pickle=True for classes (strings)
                classes = np.load(paths["classes"], allow_pickle=True)
                test_ids = np.load(paths["test_ids"])
                y_train = np.load(paths["y_train"])
                y_val = np.load(paths["y_val"])

                # Reconstruct dictionaries
                X_train = {}
                X_val = {}
                X_test = {}
                groups = list(self.tabular_groups.keys()) + ["geometry"]

                for group in groups:
                    X_train[group] = np.load(paths[f"X_train_{group}"])
                    X_val[group] = np.load(paths[f"X_val_{group}"])
                    X_test[group] = np.load(paths[f"X_test_{group}"])

                return {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_val": X_val,
                    "y_val": y_val,
                    "X_test": X_test,
                    "test_ids": test_ids,
                    "classes": classes,
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        print("Starting data preprocessing pipeline...")

        # 1. Load and Merge
        print("Loading and extracting features...")
        df_train = self._load_and_merge_geometry(TRAIN_FILE, "train")
        df_val = self._load_and_merge_geometry(VAL_FILE, "val")
        df_test = self._load_and_merge_geometry(TEST_FILE, "test")

        # 2. Process Targets
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["species"])
        y_val = le.transform(df_val["species"])
        classes = le.classes_.astype(str)  # Ensure string type for saving
        test_ids = df_test["id"].values

        # 3. Process Groups
        groups = list(self.tabular_groups.keys()) + ["geometry"]
        X_train_dict = {}
        X_val_dict = {}
        X_test_dict = {}

        for group in groups:
            print(f"Sanitizing and transforming group: {group}")

            # Extract raw data
            X_tr_raw = self._extract_group_features(df_train, group)
            X_val_raw = self._extract_group_features(df_val, group)
            X_te_raw = self._extract_group_features(df_test, group)

            # Build and Fit Pipeline (Inductive: Fit on Train only)
            pipeline = self._create_pipeline()
            X_tr_trans = pipeline.fit_transform(X_tr_raw)
            X_val_trans = pipeline.transform(X_val_raw)
            X_te_trans = pipeline.transform(X_te_raw)

            # Store in dict
            X_train_dict[group] = X_tr_trans
            X_val_dict[group] = X_val_trans
            X_test_dict[group] = X_te_trans

            # Save to cache
            np.save(paths[f"X_train_{group}"], X_tr_trans)
            np.save(paths[f"X_val_{group}"], X_val_trans)
            np.save(paths[f"X_test_{group}"], X_te_trans)

        # 4. Save Targets and Meta
        np.save(paths["y_train"], y_train)
        np.save(paths["y_val"], y_val)
        np.save(paths["test_ids"], test_ids)
        np.save(paths["classes"], classes)

        print("Preprocessing complete and cached.")

        return {
            "X_train": X_train_dict,
            "y_train": y_train,
            "X_val": X_val_dict,
            "y_val": y_val,
            "X_test": X_test_dict,
            "test_ids": test_ids,
            "classes": classes,
        }
