import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    PRECISION_TYPE,
    TABULAR_PREFIXES,
    USE_YEO_JOHNSON,
    STANDARDIZE,
    SEED,
)
from library.feature_extraction import extract_geometric_features


class LeafDataManager:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.files = {
            "X_train": os.path.join(self.cache_dir, "X_train_processed.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train_processed.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val_processed.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val_processed.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test_processed.npy"),
            "test_ids": os.path.join(self.cache_dir, "test_ids.npy"),
            "classes": os.path.join(self.cache_dir, "classes.npy"),
        }

    def _get_tabular_columns(self, df):
        """Identifies tabular feature columns based on prefixes."""
        cols = []
        for col in df.columns:
            for prefix in TABULAR_PREFIXES:
                if col.startswith(prefix):
                    cols.append(col)
                    break
        return cols

    def _prepare_split(self, metadata_path, split_name, load_cached_features=True):
        """
        Loads metadata, extracts/loads geometric features, and merges them.
        Returns features DataFrame and target/ID series.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # 1. Tabular Features
        tabular_cols = self._get_tabular_columns(df_meta)
        df_tabular = df_meta[tabular_cols].copy()

        # 2. Visual/Geometric Features
        # This function handles its own caching of the raw extraction result
        df_visual = extract_geometric_features(
            df_meta, split_name, load_cached_data=load_cached_features
        )

        # 3. Merge
        # Reset indices to ensure alignment before concat
        df_tabular.reset_index(drop=True, inplace=True)
        df_visual.reset_index(drop=True, inplace=True)

        X = pd.concat([df_tabular, df_visual], axis=1)

        # 4. Enforce Alphanumeric Column Ordering
        X = X.reindex(sorted(X.columns), axis=1)

        # 5. Extract Targets/IDs
        y = None
        if "species" in df_meta.columns:
            y = df_meta["species"].values

        ids = df_meta["id"].values

        return X, y, ids

    def load_data(self, load_cached_data=True):
        """
        Main method to load, process, and return data.
        """
        # 1. Try loading fully processed data from cache
        if load_cached_data:
            all_exist = all(os.path.exists(p) for p in self.files.values())
            if all_exist:
                print("Loading processed data from cache...")
                try:
                    X_train = np.load(self.files["X_train"])
                    y_train = np.load(self.files["y_train"])
                    X_val = np.load(self.files["X_val"])
                    y_val = np.load(self.files["y_val"])
                    X_test = np.load(self.files["X_test"])
                    test_ids = np.load(self.files["test_ids"])
                    classes = np.load(self.files["classes"], allow_pickle=True)
                    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
                except Exception as e:
                    print(f"Error loading processed cache: {e}. Recomputing...")

        print("Processing data from scratch...")

        # 2. Load Raw Splits & Merge Features
        # We pass load_cached_features=load_cached_data to the feature extractor
        # so it can use its own intermediate cache if available.
        X_train_df, y_train_raw, _ = self._prepare_split(
            TRAIN_METADATA_PATH, "train", load_cached_data
        )
        X_val_df, y_val_raw, _ = self._prepare_split(
            VAL_METADATA_PATH, "val", load_cached_data
        )
        X_test_df, _, test_ids = self._prepare_split(
            TEST_METADATA_PATH, "test", load_cached_data
        )

        # 3. Encode Targets
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_val = le.transform(y_val_raw)
        classes = le.classes_

        # 4. Convert to float64 (Double Precision)
        X_train = X_train_df.values.astype(PRECISION_TYPE)
        X_val = X_val_df.values.astype(PRECISION_TYPE)
        X_test = X_test_df.values.astype(PRECISION_TYPE)

        # 5. Preprocessing Pipeline
        # Fit ONLY on Train, Transform All

        # A. Yeo-Johnson Power Transformation
        if USE_YEO_JOHNSON:
            print("Applying Yeo-Johnson Power Transformation...")
            # standardize=False because we do it explicitly next
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            X_train = pt.fit_transform(X_train)
            X_val = pt.transform(X_val)
            X_test = pt.transform(X_test)

        # B. Standard Scaling
        if STANDARDIZE:
            print("Applying Standard Scaling...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        # 6. Save to Cache
        print("Saving processed data to cache...")
        np.save(self.files["X_train"], X_train)
        np.save(self.files["y_train"], y_train)
        np.save(self.files["X_val"], X_val)
        np.save(self.files["y_val"], y_val)
        np.save(self.files["X_test"], X_test)
        np.save(self.files["test_ids"], test_ids)
        np.save(self.files["classes"], classes)

        return X_train, y_train, X_val, y_val, X_test, test_ids, classes
