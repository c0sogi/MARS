import os
import json
import hashlib
import pandas as pd
import numpy as np
from library.config import Config
from library.feature_generators import FeatureGenerator
from library.utils import set_seed


class DataBuilder:
    """
    Orchestrates the data preparation pipeline for the Dual-Stream GBDT.
    Implements hash-based cache invalidation and random undersampling.
    """

    def __init__(self):
        self.feature_generator = FeatureGenerator()
        self.working_dir = Config.WORKING_DIR
        self.hash_file_path = os.path.join(self.working_dir, "data_config_hash.json")

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _compute_config_hash(self):
        """
        Computes a deterministic hash of the current configuration.
        Used to detect changes in features, window sizes, or sampling ratios.
        """
        config_state = {
            "FEATURES_STREAM_A": sorted(Config.FEATURES_STREAM_A),
            "FEATURES_STREAM_B": sorted(Config.FEATURES_STREAM_B),
            "WINDOW_SIZE_MICRO": Config.WINDOW_SIZE_MICRO,
            "WINDOW_SIZE_MACRO": Config.WINDOW_SIZE_MACRO,
            "NEGATIVE_RATIO": Config.NEGATIVE_RATIO,
            "SEED": Config.SEED,
        }
        # Serialize with sort_keys=True for determinism
        config_str = json.dumps(config_state, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def _check_cache_validity(self):
        """
        Verifies if the cached data corresponds to the current configuration.
        Returns:
            is_valid (bool): True if cache matches current config.
            current_hash (str): The calculated hash of the current config.
        """
        current_hash = self._compute_config_hash()

        if not os.path.exists(self.hash_file_path):
            return False, current_hash

        try:
            with open(self.hash_file_path, "r") as f:
                data = json.load(f)
                saved_hash = data.get("hash", "")

            if saved_hash == current_hash:
                return True, current_hash
            else:
                return False, current_hash
        except Exception:
            # If file is corrupt or unreadable, assume invalid
            return False, current_hash

    def _update_cache_hash(self, current_hash):
        """Updates the hash file with the new configuration hash."""
        try:
            with open(self.hash_file_path, "w") as f:
                json.dump({"hash": current_hash}, f)
        except Exception as e:
            print(f"Warning: Failed to update cache hash file: {e}")

    def _undersample_stream(self, stream_data):
        """
        Applies random undersampling to the negative class (0) to achieve
        the target negative:positive ratio defined in Config.
        """
        X = stream_data["X"]
        y = stream_data["y"]
        ids = stream_data["ids"]

        # Basic validation
        if len(X) != len(y):
            raise ValueError(
                f"Feature length {len(X)} does not match label length {len(y)}"
            )

        # Identify indices for positive and negative classes
        # y is a numpy array
        indices = np.arange(len(y))
        pos_indices = indices[y == 1]
        neg_indices = indices[y == 0]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Calculate target number of negatives
        n_neg_keep = int(n_pos * Config.NEGATIVE_RATIO)

        if n_neg > n_neg_keep:
            # Sample negatives
            np.random.seed(Config.SEED)
            neg_indices_sampled = np.random.choice(
                neg_indices, size=n_neg_keep, replace=False
            )

            # Combine
            keep_indices = np.concatenate([pos_indices, neg_indices_sampled])

            # Shuffle to mix classes
            np.random.shuffle(keep_indices)

            # Subset data
            # X is a DataFrame, use iloc
            X_sampled = X.iloc[keep_indices].reset_index(drop=True)
            y_sampled = y[keep_indices]
            ids_sampled = ids[keep_indices]

            return {"X": X_sampled, "y": y_sampled, "ids": ids_sampled}
        else:
            # If we don't have enough negatives to reach the ratio, keep all
            return stream_data

    def get_stream_data(self, split="train", load_cached_data=True):
        """
        Main entry point to retrieve processed data for Stream A and Stream B.

        Args:
            split (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): If True, attempts to load from disk.
                                     If False or cache invalid, regenerates.

        Returns:
            dict: Dictionary with keys 'stream_a' and 'stream_b', each containing
                  {'X': DataFrame, 'y': array, 'ids': array}.
        """
        set_seed(Config.SEED)

        # 1. Check Cache Validity
        is_cache_valid, current_hash = self._check_cache_validity()

        # Determine if we can load from cache
        # We load if user requested it AND the config hasn't changed
        should_load = load_cached_data and is_cache_valid

        if not should_load:
            if not is_cache_valid and load_cached_data:
                print(
                    "Configuration change detected. Invalidating cache and regenerating features..."
                )
            elif not load_cached_data:
                print(f"Force reload requested. Regenerating features for {split}...")

        # 2. Generate or Load Features
        # We delegate to FeatureGenerator.
        # If should_load is False, FeatureGenerator will ignore existing parquet files and rebuild.
        data = self.feature_generator.generate_features(
            split=split, load_cached_data=should_load
        )

        # If we regenerated (or if cache was invalid), update the hash
        if not is_cache_valid:
            self._update_cache_hash(current_hash)

        # 3. Apply Undersampling (Only for Training)
        if split == "train":
            print(
                f"Applying undersampling (Ratio 1:{Config.NEGATIVE_RATIO}) to training data..."
            )

            # Process Stream A
            orig_len_a = len(data["stream_a"]["y"])
            data["stream_a"] = self._undersample_stream(data["stream_a"])
            new_len_a = len(data["stream_a"]["y"])

            # Process Stream B
            orig_len_b = len(data["stream_b"]["y"])
            data["stream_b"] = self._undersample_stream(data["stream_b"])
            new_len_b = len(data["stream_b"]["y"])

            print(f"Stream A: Reduced from {orig_len_a} to {new_len_a} samples.")
            print(f"Stream B: Reduced from {orig_len_b} to {new_len_b} samples.")

        return data
