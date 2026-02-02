import os
import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.utils import set_seed, get_cache_dir
from library.features import get_dataset


class DataPipeline:
    def __init__(self, debug: bool = False, seed: int = 42):
        """
        Initializes the DataPipeline.

        Args:
            debug (bool): If True, uses a subset of the data.
            seed (int): Random seed for reproducibility.
        """
        self.debug = debug
        self.seed = seed
        set_seed(seed)
        self.cache_dir = get_cache_dir()

    def run(self, load_cached_data: bool = True):
        """
        Executes the full data pipeline: loading, cleaning, transforming, and caching.

        Args:
            load_cached_data (bool): If True, attempts to load processed arrays from disk.

        Returns:
            dict: A dictionary containing the processed datasets:
                - 'train': (X_train, y_train, ids_train)
                - 'val': (X_val, y_val, ids_val)
                - 'test': (X_test, ids_test)
                - 'classes': Array of class names corresponding to encoded targets.
        """
        # Define paths for cached processed arrays
        cache_files = {
            "X_train": os.path.join(self.cache_dir, "X_train_processed.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train_processed.npy"),
            "ids_train": os.path.join(self.cache_dir, "ids_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val_processed.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val_processed.npy"),
            "ids_val": os.path.join(self.cache_dir, "ids_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test_processed.npy"),
            "ids_test": os.path.join(self.cache_dir, "ids_test.npy"),
            "classes": os.path.join(self.cache_dir, "classes.npy"),
        }

        # Attempt to load from cache
        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            logging.info("Loading processed data from cache...")
            try:
                data = {
                    "train": (
                        np.load(cache_files["X_train"]),
                        np.load(cache_files["y_train"]),
                        np.load(cache_files["ids_train"]),
                    ),
                    "val": (
                        np.load(cache_files["X_val"]),
                        np.load(cache_files["y_val"]),
                        np.load(cache_files["ids_val"]),
                    ),
                    "test": (
                        np.load(cache_files["X_test"]),
                        np.load(cache_files["ids_test"]),
                    ),
                    "classes": np.load(cache_files["classes"], allow_pickle=True),
                }
                logging.info("Successfully loaded processed data from cache.")
                return data
            except Exception as e:
                logging.warning(
                    f"Failed to load processed cache: {e}. Recomputing from scratch..."
                )

        # 1. Load DataFrames (with geometric feature extraction)
        logging.info("Loading datasets and extracting features...")
        df_train = get_dataset(
            "train", debug=self.debug, load_cached_data=load_cached_data
        )
        df_val = get_dataset("val", debug=self.debug, load_cached_data=load_cached_data)
        df_test = get_dataset(
            "test", debug=self.debug, load_cached_data=load_cached_data
        )

        # 2. Extract IDs and Targets
        target_col = "species"
        id_col = "id"

        ids_train = df_train[id_col].values
        ids_val = df_val[id_col].values
        ids_test = df_test[id_col].values

        # Encode Targets
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[target_col].values)
        y_val = le.transform(df_val[target_col].values)
        classes = le.classes_

        # 3. Prepare Feature Matrices (Float64)
        # Drop metadata columns to isolate features
        drop_cols_train = [id_col, target_col]
        drop_cols_test = [id_col]

        X_train = df_train.drop(columns=drop_cols_train).values.astype(np.float64)
        X_val = df_val.drop(columns=drop_cols_train).values.astype(np.float64)
        X_test = df_test.drop(columns=drop_cols_test).values.astype(np.float64)

        logging.info(f"Initial Feature Shape: {X_train.shape}")

        # 4. Pipeline Sanitization: Variance Thresholding
        # Remove constant features to prevent scaler explosion
        logging.info("Applying VarianceThreshold(threshold=0)...")
        vt = VarianceThreshold(threshold=0)
        X_train = vt.fit_transform(X_train)
        X_val = vt.transform(X_val)
        X_test = vt.transform(X_test)
        logging.info(f"Shape after VarianceThreshold: {X_train.shape}")

        # 5. Inductive Transformation: PowerTransformer (Yeo-Johnson)
        # Stabilize variance; standardize=False to allow StandardScaler to handle scaling next
        logging.info("Applying PowerTransformer(method='yeo-johnson')...")
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        X_train = pt.fit_transform(X_train)
        X_val = pt.transform(X_val)
        X_test = pt.transform(X_test)

        # 6. Scaling: StandardScaler
        # Center and scale to unit variance
        logging.info("Applying StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # 7. Cache Results
        logging.info("Caching processed arrays...")
        try:
            np.save(cache_files["X_train"], X_train)
            np.save(cache_files["y_train"], y_train)
            np.save(cache_files["ids_train"], ids_train)
            np.save(cache_files["X_val"], X_val)
            np.save(cache_files["y_val"], y_val)
            np.save(cache_files["ids_val"], ids_val)
            np.save(cache_files["X_test"], X_test)
            np.save(cache_files["ids_test"], ids_test)
            np.save(cache_files["classes"], classes)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

        return {
            "train": (X_train, y_train, ids_train),
            "val": (X_val, y_val, ids_val),
            "test": (X_test, ids_test),
            "classes": classes,
        }
