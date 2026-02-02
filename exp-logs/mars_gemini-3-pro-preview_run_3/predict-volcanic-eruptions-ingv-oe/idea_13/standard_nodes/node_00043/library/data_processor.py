import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library.config import INPUT_DIR, METADATA_DIR, CACHE_DIR, N_JOBS
from library.feature_extraction import process_segment


def _process_wrapper(file_path, segment_id, target=None):
    """
    Helper function to process a single sensor file.
    Reads the CSV and applies the feature extraction pipeline.
    """
    try:
        full_path = os.path.join(INPUT_DIR, file_path)
        # Load data as float32 to handle potential NaNs and optimize memory
        df = pd.read_csv(full_path, dtype="float32")

        # Apply the imported feature extraction logic
        features = process_segment(df)

        return segment_id, features, target
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return segment_id, None, target


class DatasetBuilder:
    """
    Manages the loading of sensor CSVs, parallel feature extraction, and caching.
    """

    def __init__(self, metadata_dir=METADATA_DIR, cache_dir=CACHE_DIR, n_jobs=N_JOBS):
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        self.n_jobs = n_jobs
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_metadata(self, split):
        """Loads the metadata CSV for the specified split."""
        path = os.path.join(self.metadata_dir, f"{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_csv(path)

    def _build_dataset(self, split, load_cached_data=True, debug_size=None):
        """
        Core logic to build or load the dataset for a given split.
        Handles caching via Parquet and parallel processing via joblib.
        """
        # Construct a unique cache filename based on split and debug status
        cache_filename = f"{split}_features.parquet"
        if debug_size:
            cache_filename = f"debug_{debug_size}_{cache_filename}"

        cache_path = os.path.join(self.cache_dir, cache_filename)

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} features from {cache_path}...")
            df = pd.read_parquet(cache_path)

            # Extract components
            feature_cols = [c for c in df.columns if c.startswith("f_")]
            X = df[feature_cols].values.astype(np.float32)
            segment_ids = df["segment_id"].values

            y = None
            if "time_to_eruption" in df.columns:
                y = df["time_to_eruption"].values.astype(np.float32)

            # For test set, y remains None if not in columns
            return X, y, segment_ids

        # 2. Generate features from scratch
        print(f"Generating {split} features from scratch...")
        meta_df = self._load_metadata(split)

        if debug_size:
            print(f"Debug mode: processing first {debug_size} samples.")
            meta_df = meta_df.iloc[:debug_size]

        # Prepare tasks for parallel execution
        tasks = []
        for _, row in meta_df.iterrows():
            # Target is NaN for test set
            target = row["time_to_eruption"] if "time_to_eruption" in row else np.nan
            tasks.append((row["file_path"], row["segment_id"], target))

        # Execute parallel processing
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_process_wrapper)(fp, sid, tgt) for fp, sid, tgt in tasks
        )

        # Filter out any failed files (where features is None)
        valid_results = [r for r in results if r[1] is not None]
        if len(valid_results) < len(results):
            print(
                f"Warning: {len(results) - len(valid_results)} files failed to process."
            )

        if not valid_results:
            raise ValueError("No files were successfully processed.")

        # Unpack results
        segment_ids = np.array([r[0] for r in valid_results])
        X_list = [r[1] for r in valid_results]
        y_list = [r[2] for r in valid_results]

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)

        # 3. Save to cache
        print(f"Saving features to {cache_path}...")
        feature_cols = [f"f_{i}" for i in range(X.shape[1])]
        cache_df = pd.DataFrame(X, columns=feature_cols)
        cache_df["segment_id"] = segment_ids

        # Only save target if it contains valid data (Train/Val)
        # For test set, y contains NaNs, so we skip saving/loading it as a target column usually,
        # or we can save it. Here we only save if it's the train/val split logic.
        if split in ["train", "val"]:
            cache_df["time_to_eruption"] = y

        cache_df.to_parquet(cache_path)

        # Return None for y if it's the test set (all NaNs)
        if split == "test":
            y = None

        return X, y, segment_ids

    def get_train_data(self, load_cached_data=True, debug_size=None):
        """Wrapper to get the training dataset."""
        return self._build_dataset("train", load_cached_data, debug_size)

    def get_val_data(self, load_cached_data=True, debug_size=None):
        """Wrapper to get the validation dataset."""
        return self._build_dataset("val", load_cached_data, debug_size)

    def get_test_data(self, load_cached_data=True, debug_size=None):
        """Wrapper to get the test dataset."""
        return self._build_dataset("test", load_cached_data, debug_size)
