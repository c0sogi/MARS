import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder

from library.config import INPUT_DIR, METADATA_DIR, CACHE_DIR, CONFIG_HASH_DICT, SEED
from library.features import batch_extract_features
from library.utils import set_seed, generate_config_hash


class DataManager:
    """
    Handles data loading, feature merging, deterministic caching, and
    high-precision preprocessing for the leaf classification task.
    """

    def __init__(
        self, input_dir=INPUT_DIR, metadata_dir=METADATA_DIR, cache_dir=CACHE_DIR
    ):
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set global seeds
        set_seed(SEED)

    def _load_raw_merged(self, mode, load_cached_data=True):
        """
        Internal helper to load metadata, extract/load geometric features,
        merge them with tabular features, and sort columns deterministically.
        """
        # 1. Load Metadata
        meta_path = os.path.join(self.metadata_dir, f"{mode}.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # 2. Get Geometric Features
        # Uses the library function which handles caching of the extraction step
        df_geo = batch_extract_features(
            df_meta, self.input_dir, load_cached_data=load_cached_data
        )

        # 3. Merge Geometric Features with Metadata (Tabular Features)
        # Perform left join on 'id' to ensure we keep all metadata rows
        df_merged = pd.merge(df_meta, df_geo, on="id", how="left")

        # 4. Deterministic Column Ordering
        # Identify feature columns (exclude non-feature columns)
        exclude_cols = ["id", "species", "file_path"]
        feature_cols = [c for c in df_merged.columns if c not in exclude_cols]
        feature_cols.sort()  # Alphanumeric sort

        # Reconstruct DataFrame with specific order: id, species (if present), features
        final_cols = ["id"]
        if "species" in df_merged.columns:
            final_cols.append("species")
        final_cols.extend(feature_cols)

        return df_merged[final_cols], feature_cols

    def get_processed_data(self, load_cached_data=True):
        """
        Orchestrates the loading and preprocessing pipeline.
        Checks for cached processed arrays (npz). If missing, computes from scratch:
          1. Loads merged raw data for Train, Val, Test.
          2. Fits Yeo-Johnson and StandardScaler on Train.
          3. Transforms Train, Val, Test.
          4. Caches the result.

        Returns:
            tuple: (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test, classes)
        """
        # Generate a unique hash for the processed dataset configuration
        # We hash the configuration dictionary to ensure version control
        config_hash = generate_config_hash(CONFIG_HASH_DICT)
        cache_filename = f"processed_dataset_{config_hash}.npz"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading processed data from {cache_path}")
            try:
                data = np.load(cache_path, allow_pickle=True)
                return (
                    data["X_train"],
                    data["y_train"],
                    data["ids_train"],
                    data["X_val"],
                    data["y_val"],
                    data["ids_val"],
                    data["X_test"],
                    data["ids_test"],
                    data["classes"],
                )
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        print("Processing data from scratch (High-Precision Pipeline)...")

        # Load raw merged dataframes
        df_train, feat_cols = self._load_raw_merged("train", load_cached_data)
        df_val, _ = self._load_raw_merged("val", load_cached_data)
        df_test, _ = self._load_raw_merged("test", load_cached_data)

        print(f"Total Features: {len(feat_cols)}")

        # Extract IDs
        ids_train = df_train["id"].values
        ids_val = df_val["id"].values
        ids_test = df_test["id"].values

        # Extract Targets and Encode
        le = LabelEncoder()
        y_train_raw = df_train["species"].values
        y_val_raw = df_val["species"].values

        y_train_enc = le.fit_transform(y_train_raw)
        y_val_enc = le.transform(y_val_raw)
        classes = le.classes_

        # Extract Features (Ensure float64)
        X_train = df_train[feat_cols].values.astype(np.float64)
        X_val = df_val[feat_cols].values.astype(np.float64)
        X_test = df_test[feat_cols].values.astype(np.float64)

        # Inductive Preprocessing Pipeline
        # Fit ONLY on training data

        # Step A: Yeo-Johnson Power Transformation
        # standardize=False because we will apply StandardScaler explicitly next
        print("Applying Yeo-Johnson Power Transformation...")
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        X_train_pt = pt.fit_transform(X_train)
        X_val_pt = pt.transform(X_val)
        X_test_pt = pt.transform(X_test)

        # Step B: Standard Scaling
        print("Applying Standard Scaling...")
        ss = StandardScaler()
        X_train_scaled = ss.fit_transform(X_train_pt)
        X_val_scaled = ss.transform(X_val_pt)
        X_test_scaled = ss.transform(X_test_pt)

        # 3. Save to Cache
        print(f"Saving processed dataset to {cache_path}")
        np.savez(
            cache_path,
            X_train=X_train_scaled,
            y_train=y_train_enc,
            ids_train=ids_train,
            X_val=X_val_scaled,
            y_val=y_val_enc,
            ids_val=ids_val,
            X_test=X_test_scaled,
            ids_test=ids_test,
            classes=classes,
        )

        return (
            X_train_scaled,
            y_train_enc,
            ids_train,
            X_val_scaled,
            y_val_enc,
            ids_val,
            X_test_scaled,
            ids_test,
            classes,
        )
