import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from library.config import Config


class FeatureScaler:
    """
    Handles feature normalization using QuantileTransformer.
    Includes caching mechanisms to persist scaled data to disk.
    """

    def __init__(self):
        self.config = Config
        # Use QuantileTransformer to gaussianize features (Cite solution_lesson_node_00001)
        # n_quantiles set to 500 (must be < n_samples which is ~700)
        self.scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=500,
            random_state=Config.RANDOM_SEED,
        )

    def scale_features(self, X_train, X_val, X_test, load_cached_data=True):
        """
        Fits scaler on training data and transforms train, val, and test sets.
        Uses caching to store/retrieve processed numpy arrays.

        Args:
            X_train (np.ndarray): Training feature matrix.
            X_val (np.ndarray): Validation feature matrix.
            X_test (np.ndarray): Test feature matrix.
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        # Ensure working directory exists
        cache_dir = self.config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        # Define cache file paths
        # Changed filenames to avoid loading stale StandardScaler data
        train_path = os.path.join(cache_dir, "X_train_quantile.npy")
        val_path = os.path.join(cache_dir, "X_val_quantile.npy")
        test_path = os.path.join(cache_dir, "X_test_quantile.npy")

        # Check if all cache files exist
        cache_exists = (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        )

        if load_cached_data and cache_exists:
            print("Loading scaled features from cache...")
            X_train_scaled = np.load(train_path)
            X_val_scaled = np.load(val_path)
            X_test_scaled = np.load(test_path)
        else:
            print("Scaling features (QuantileTransformer)...")
            # Fit only on training data to prevent data leakage
            self.scaler.fit(X_train)

            # Transform all datasets
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)

            # Save to cache
            print(f"Saving scaled features to {cache_dir}...")
            np.save(train_path, X_train_scaled)
            np.save(val_path, X_val_scaled)
            np.save(test_path, X_test_scaled)

        return X_train_scaled, X_val_scaled, X_test_scaled
