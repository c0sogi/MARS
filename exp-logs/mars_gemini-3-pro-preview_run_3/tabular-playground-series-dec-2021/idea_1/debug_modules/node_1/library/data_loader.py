import os
import pandas as pd
import numpy as np
from library.config import Config


class ForestDataLoader:
    """
    Data loader for the Forest Cover Type prediction task.
    Handles loading from Parquet metadata, preprocessing, and caching.
    """

    def __init__(self):
        self.config = Config
        # Ensure working directory exists for caching as per requirements
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def get_data(self, split="train", load_cached_data=True):
        """
        Retrieves the features (X) and target (y) for a given split.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load processed data from cache.

        Returns:
            tuple: (X, y)
                X (pd.DataFrame): Feature matrix.
                y (np.ndarray or None): Target vector (0-indexed) or None for test split.
        """
        # Define cache paths
        cache_x_path = os.path.join(self.config.WORKING_DIR, f"{split}_X.parquet")
        cache_y_path = os.path.join(self.config.WORKING_DIR, f"{split}_y.npy")

        # 1. Try to load from cache
        if load_cached_data:
            # Check if X exists. For test, y is not needed. For others, y must exist.
            x_exists = os.path.exists(cache_x_path)
            y_exists = os.path.exists(cache_y_path)

            if x_exists and (split == "test" or y_exists):
                X = pd.read_parquet(cache_x_path)
                if split == "test":
                    y = None
                else:
                    y = np.load(cache_y_path)
                return X, y

        # 2. Compute from scratch
        df = self._load_raw_file(split)
        X, y = self._prepare_inputs(df, split)

        # 3. Save to cache
        X.to_parquet(cache_x_path, index=False)
        if y is not None:
            np.save(cache_y_path, y)

        return X, y

    def _load_raw_file(self, split):
        """Loads the raw parquet file based on split configuration."""
        if split == "train":
            path = self.config.TRAIN_DATA_PATH
        elif split == "val":
            path = self.config.VAL_DATA_PATH
        elif split == "test":
            path = self.config.TEST_DATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")

        df = pd.read_parquet(path)

        # Handle Debugging Subsample
        if self.config.DEBUG:
            df = df.iloc[: self.config.DEBUG_SAMPLE_SIZE].copy()

        return df

    def _prepare_inputs(self, df, split):
        """Separates features and target, drops ID, and normalizes target."""
        # Drop ID column if present
        if self.config.ID_COL in df.columns:
            df = df.drop(columns=[self.config.ID_COL])

        y = None
        if split != "test":
            if self.config.TARGET_COL not in df.columns:
                raise ValueError(
                    f"Target column '{self.config.TARGET_COL}' missing in {split} data."
                )

            # Extract target
            y = df[self.config.TARGET_COL].values

            # Convert 1-based labels (1-7) to 0-based (0-6) for XGBoost
            # We assume the input is 1-based as per standard CoverType dataset
            y = y - 1

            # Drop target from features
            X = df.drop(columns=[self.config.TARGET_COL])
        else:
            # For test set, ensure target is removed if accidentally present
            X = df.copy()
            if self.config.TARGET_COL in X.columns:
                X = X.drop(columns=[self.config.TARGET_COL])

        return X, y
