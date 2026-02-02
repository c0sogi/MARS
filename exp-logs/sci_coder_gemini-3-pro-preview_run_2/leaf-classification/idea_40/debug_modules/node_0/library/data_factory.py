import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    FLOAT_PRECISION,
)
from library.features import process_image_batch


class DataFactory:
    """
    Handles data ingestion, view construction, and caching for the Leaf Classification task.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the DataFactory.

        Args:
            load_cached_data (bool): If True, attempts to load processed matrices from cache.
        """
        self.load_cached_data = load_cached_data

        # 1. Load Metadata
        # We assume metadata files exist as per the task description
        self.train_df = pd.read_csv(TRAIN_METADATA_PATH)
        self.val_df = pd.read_csv(VAL_METADATA_PATH)
        self.test_df = pd.read_csv(TEST_METADATA_PATH)

        # 2. Setup Label Encoder
        # Fit on the union of train and validation species to ensure all classes are covered
        # and the mapping is consistent across splits.
        train_species = self.train_df["species"].unique()
        val_species = self.val_df["species"].unique()
        all_species = sorted(list(set(train_species) | set(val_species)))

        self.le = LabelEncoder()
        self.le.fit(all_species)
        self.classes = self.le.classes_

        # 3. Define Base Feature Columns
        # Strictly define the order of the 192 provided features
        self.margin_cols = [f"margin_{i}" for i in range(1, 65)]
        self.shape_cols = [f"shape_{i}" for i in range(1, 65)]
        self.texture_cols = [f"texture_{i}" for i in range(1, 65)]
        self.base_feature_cols = self.margin_cols + self.shape_cols + self.texture_cols

        # Ensure working directory exists for caching
        os.makedirs(WORKING_DIR, exist_ok=True)

    def get_classes(self):
        """Returns the list of class names."""
        return self.classes

    def get_data(self, split_name, view_name):
        """
        Retrieves the feature matrix and targets/ids for a specific split and view.
        Implements caching to speed up subsequent calls.

        Args:
            split_name (str): One of 'train', 'val', 'test', 'train_full'.
                              'train_full' combines 'train' and 'val'.
            view_name (str): One of 'global', 'combined'.
                             'global' = 192 provided features.
                             'combined' = 192 features + 10 extracted morphometrics.

        Returns:
            tuple:
                - If split_name == 'test': (X, ids)
                - Otherwise: (X, y)
                X is a numpy array of type FLOAT_PRECISION.
                y is a numpy array of integer encoded labels.
                ids is a numpy array of image identifiers.
        """
        # 1. Determine Source DataFrame
        if split_name == "train":
            df = self.train_df
        elif split_name == "val":
            df = self.val_df
        elif split_name == "test":
            df = self.test_df
        elif split_name == "train_full":
            # Concatenate train and val for the final retraining phase
            df = pd.concat([self.train_df, self.val_df], axis=0).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        # 2. Define Cache Paths
        # We cache the final X matrix for the specific split/view combination
        cache_X_path = os.path.join(WORKING_DIR, f"X_{split_name}_{view_name}.npy")
        cache_y_path = os.path.join(WORKING_DIR, f"y_{split_name}.npy")
        cache_ids_path = os.path.join(WORKING_DIR, f"ids_{split_name}.npy")

        # 3. Try Loading from Cache
        if self.load_cached_data:
            if split_name == "test":
                if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
                    print(
                        f"[DataFactory] Loading cached {split_name} data ({view_name})..."
                    )
                    return np.load(cache_X_path), np.load(cache_ids_path)
            else:
                if os.path.exists(cache_X_path) and os.path.exists(cache_y_path):
                    print(
                        f"[DataFactory] Loading cached {split_name} data ({view_name})..."
                    )
                    return np.load(cache_X_path), np.load(cache_y_path)

        # 4. Compute Data (Cache Miss)
        print(f"[DataFactory] Constructing {split_name} data ({view_name})...")

        # A. Extract Base Features (Global View)
        # Ensure strict column ordering and precision
        X = df[self.base_feature_cols].values.astype(FLOAT_PRECISION)

        # B. If Combined View, Add Morphometrics
        if view_name == "combined":
            # process_image_batch handles its own caching of the morphometric features
            morph_feats = process_image_batch(
                df, dataset_name=split_name, load_cached_data=self.load_cached_data
            )
            # Concatenate base features with morphometrics
            X = np.hstack([X, morph_feats])

        # C. Handle Targets / IDs and Save Cache
        if split_name == "test":
            ids = df["id"].values
            np.save(cache_X_path, X)
            np.save(cache_ids_path, ids)
            return X, ids
        else:
            y = self.le.transform(df["species"].values)
            np.save(cache_X_path, X)
            np.save(cache_y_path, y)
            return X, y
