import os
import numpy as np
from library.feature_extraction import run_feature_extraction


class DataPreprocessor:
    """
    Orchestrates the assembly of features from various sources (Text, Metadata).
    Handles concatenation and caching of the final feature matrices.
    """

    def __init__(self, k_neighbors=50):
        # k_neighbors is kept for compatibility but ignored (Cite Lesson 24)
        self.k_neighbors = k_neighbors
        self.cache_dir = "./working/idea_9/"
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_and_load_data(self, load_cached_data=True, debug_sample_size=None):
        """
        Loads features and assembles the final dataset.

        Args:
            load_cached_data (bool): Whether to load from cache if available.
            debug_sample_size (int, optional): Number of samples for debugging.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
        """
        # Define cache file paths for the assembled data
        files = {
            "X_train": os.path.join(self.cache_dir, "X_train_assembled.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train_assembled.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val_assembled.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val_assembled.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test_assembled.npy"),
            "test_ids": os.path.join(self.cache_dir, "test_ids_assembled.npy"),
        }

        # 1. Try Loading from Cache
        if load_cached_data and all(os.path.exists(f) for f in files.values()):
            print("Loading assembled features from cache...")
            try:
                X_train = np.load(files["X_train"])
                y_train = np.load(files["y_train"])
                X_val = np.load(files["X_val"])
                y_val = np.load(files["y_val"])
                X_test = np.load(files["X_test"])
                test_ids = np.load(files["test_ids"], allow_pickle=True)
                return X_train, y_train, X_val, y_val, X_test, test_ids
            except Exception as e:
                print(f"Failed to load assembled cache: {e}. Recomputing...")

        print("Assembling features from scratch...")

        # 2. Retrieve Component Features
        # This handles text embeddings (L2 normalized) and metadata (RankGauss scaled)
        feats = run_feature_extraction(
            load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
        )

        X_text_train = feats["X_text_train"]
        X_text_val = feats["X_text_val"]
        X_text_test = feats["X_text_test"]
        X_meta_train = feats["X_meta_train"]
        X_meta_val = feats["X_meta_val"]
        X_meta_test = feats["X_meta_test"]
        y_train = feats["y_train"]
        y_val = feats["y_val"]
        test_ids = feats["test_ids"]

        # Cite Lesson 24: Avoid compressing high-dimensional semantic embeddings into scalar proximity features.
        # KNN feature generation is removed.

        # 4. Concatenate Features
        # Order: [Text Embeddings, Metadata]
        print("Concatenating feature sets...")
        X_train = np.hstack([X_text_train, X_meta_train])
        X_val = np.hstack([X_text_val, X_meta_val])
        X_test = np.hstack([X_text_test, X_meta_test])

        # 5. Save to Cache
        print("Saving assembled features to cache...")
        np.save(files["X_train"], X_train)
        np.save(files["y_train"], y_train)
        np.save(files["X_val"], X_val)
        np.save(files["y_val"], y_val)
        np.save(files["X_test"], X_test)
        np.save(files["test_ids"], test_ids)

        return X_train, y_train, X_val, y_val, X_test, test_ids
