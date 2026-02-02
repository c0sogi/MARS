import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from library.config import Config


class LeafDataLoader:
    """
    Handles loading, preprocessing, and caching of Leaf Classification data.
    """

    def __init__(self):
        self.config = Config
        self.encoder = LabelEncoder()

    def load_data(self, load_cached_data=True):
        """
        Loads data from cache or processes it from raw metadata.

        Args:
            load_cached_data (bool): If True, attempts to load from local parquet cache.

        Returns:
            dict: A dictionary containing:
                - 'train': (X_train, y_train, train_ids)
                - 'val': (X_val, y_val, val_ids)
                - 'test': (X_test, test_ids)
                - 'encoder': Fitted LabelEncoder instance
        """
        # Define cache paths
        cache_dir = self.config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        train_cache_path = os.path.join(cache_dir, "train_processed.parquet")
        val_cache_path = os.path.join(cache_dir, "val_processed.parquet")
        test_cache_path = os.path.join(cache_dir, "test_processed.parquet")
        classes_cache_path = os.path.join(cache_dir, "classes.npy")

        # Check if cache exists
        cache_exists = (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
            and os.path.exists(classes_cache_path)
        )

        if load_cached_data and cache_exists:
            print("Loading data from cache...")
            df_train = pd.read_parquet(train_cache_path)
            df_val = pd.read_parquet(val_cache_path)
            df_test = pd.read_parquet(test_cache_path)

            # Restore LabelEncoder
            self.encoder.classes_ = np.load(classes_cache_path, allow_pickle=True)

        else:
            print("Processing data from metadata...")
            # Load raw metadata
            df_train = pd.read_csv(self.config.TRAIN_DATA_PATH)
            df_val = pd.read_csv(self.config.VAL_DATA_PATH)
            df_test = pd.read_csv(self.config.TEST_DATA_PATH)

            # Fit LabelEncoder on training species
            self.encoder.fit(df_train[self.config.TARGET_COL])

            # Encode targets
            df_train["target_encoded"] = self.encoder.transform(
                df_train[self.config.TARGET_COL]
            )
            df_val["target_encoded"] = self.encoder.transform(
                df_val[self.config.TARGET_COL]
            )

            # Save to cache
            df_train.to_parquet(train_cache_path, index=False)
            df_val.to_parquet(val_cache_path, index=False)
            df_test.to_parquet(test_cache_path, index=False)
            np.save(classes_cache_path, self.encoder.classes_)

        # Extract Features, Targets, and IDs
        # Identify feature columns (exclude id, species, image_path, target_encoded)
        exclude_cols = {
            self.config.ID_COL,
            self.config.TARGET_COL,
            self.config.IMAGE_PATH_COL,
            "target_encoded",
        }
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

        # Ensure consistent column order
        feature_cols.sort()

        # Prepare Train Data
        X_train = df_train[feature_cols].values.astype(np.float32)
        y_train = df_train["target_encoded"].values.astype(np.int64)
        train_ids = df_train[self.config.ID_COL].values

        # Prepare Val Data
        X_val = df_val[feature_cols].values.astype(np.float32)
        y_val = df_val["target_encoded"].values.astype(np.int64)
        val_ids = df_val[self.config.ID_COL].values

        # Prepare Test Data
        # Test set might not have 'species' or 'target_encoded'
        test_feature_cols = [c for c in df_test.columns if c not in exclude_cols]
        test_feature_cols.sort()

        # Verify feature alignment
        if feature_cols != test_feature_cols:
            raise ValueError("Feature columns in Train and Test datasets do not match.")

        X_test = df_test[feature_cols].values.astype(np.float32)
        test_ids = df_test[self.config.ID_COL].values

        return {
            "train": (X_train, y_train, train_ids),
            "val": (X_val, y_val, val_ids),
            "test": (X_test, test_ids),
            "encoder": self.encoder,
        }
