import os
import pandas as pd
import numpy as np
from library.config import Config


class HistogramDataLoader:
    """
    Data Loader for the Histogram of Segments features.
    Handles loading raw text features, aligning with metadata, and caching.
    """

    def __init__(self):
        self.config = Config
        self.label_cols = [f"species_{i}" for i in range(self.config.NUM_SPECIES)]

    def _parse_histogram_file(self):
        """
        Parses the histogram_of_segments.txt file.
        The file has a header line with 2 columns but data lines with 101 columns.
        """
        file_path = self.config.HISTOGRAM_FILE
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Histogram file not found at {file_path}")

        # Read the file, skipping the header row because it is malformed relative to data
        # Header: rec_id,[histogram of segment features] (2 elements)
        # Data: rec_id, feat1, feat2... (101 elements)
        try:
            # We assume the file is comma-separated.
            # skiprows=1 ignores the header. header=None treats the first data row as data, not header.
            df = pd.read_csv(file_path, skiprows=1, header=None)
        except Exception as e:
            print(f"Error reading histogram file: {e}")
            return pd.DataFrame()

        # Rename columns
        # Col 0 is rec_id
        # Cols 1..100 are features
        # We verify the number of columns matches expectation (101)
        if df.shape[1] < 2:
            raise ValueError(f"Histogram file has unexpected shape: {df.shape}")

        feature_cols = [f"feat_{i}" for i in range(df.shape[1] - 1)]
        df.columns = ["rec_id"] + feature_cols

        return df

    def get_data_splits(self, load_cached_data=True, max_samples=None):
        """
        Loads training, validation, and test sets.

        Args:
            load_cached_data (bool): Whether to load from parquet cache.
            max_samples (int, optional): Limit dataset size for debugging.

        Returns:
            tuple: ((X_train, y_train), (X_val, y_val), (X_test, test_ids))
        """
        cache_path = self.config.CACHE_FILE

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            df_all = pd.read_parquet(cache_path)
        else:
            print("Processing data from scratch...")
            # 2. Load Metadata
            if not os.path.exists(self.config.TRAIN_CSV):
                raise FileNotFoundError(
                    f"Metadata not found at {self.config.TRAIN_CSV}"
                )

            df_train_meta = pd.read_csv(self.config.TRAIN_CSV)
            df_val_meta = pd.read_csv(self.config.VAL_CSV)
            df_test_meta = pd.read_csv(self.config.TEST_CSV)

            # Add split identifier
            df_train_meta["split"] = "train"
            df_val_meta["split"] = "val"
            df_test_meta["split"] = "test"

            # Concatenate metadata
            df_meta = pd.concat(
                [df_train_meta, df_val_meta, df_test_meta], ignore_index=True
            )

            # 3. Load Features
            df_features = self._parse_histogram_file()

            # 4. Merge
            # Inner join to ensure we only have records with both metadata and features
            df_all = pd.merge(df_meta, df_features, on="rec_id", how="inner")

            # 5. Save Cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df_all.to_parquet(cache_path, index=False)
            print(f"Saved processed data to {cache_path}")

        # 6. Split back into sets
        df_train = df_all[df_all["split"] == "train"].copy()
        df_val = df_all[df_all["split"] == "val"].copy()
        df_test = df_all[df_all["split"] == "test"].copy()

        # 7. Apply Debug Limits
        if max_samples is not None:
            df_train = df_train.iloc[:max_samples]
            df_val = df_val.iloc[:max_samples]
            df_test = df_test.iloc[:max_samples]

        # 8. Extract Features and Labels
        # Identify feature columns (those starting with 'feat_')
        feat_cols = [c for c in df_train.columns if c.startswith("feat_")]

        # Ensure we have features
        if not feat_cols:
            raise ValueError("No feature columns found in processed data.")

        X_train = df_train[feat_cols].values.astype(np.float32)
        y_train = df_train[self.label_cols].values.astype(np.int32)

        X_val = df_val[feat_cols].values.astype(np.float32)
        y_val = df_val[self.label_cols].values.astype(np.int32)

        X_test = df_test[feat_cols].values.astype(np.float32)
        # For test, we need rec_id to construct the submission IDs
        test_ids = df_test["rec_id"].values.astype(np.int32)

        return (X_train, y_train), (X_val, y_val), (X_test, test_ids)
