import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import (
    USE_POWER_TRANSFORM,
    POWER_TRANSFORM_METHOD,
    USE_STANDARD_SCALING,
    CACHE_TRAIN_PATH,
    CACHE_VAL_PATH,
    CACHE_TEST_PATH,
    WORKING_DIR,
    SEED,
)
from library.utils import set_seed


class GlobalPreprocessor:
    """
    Handles the global preprocessing pipeline required for the Feature-Bagged LDA Ensemble.

    Pipeline:
    1. Power Transformation (Yeo-Johnson): To enforce multivariate Gaussian distribution.
    2. Standard Scaling: To ensure numerical stability and zero-mean/unit-variance.

    Includes caching logic to store transformed arrays in the working directory.
    """

    def __init__(self):
        self.pt = None
        self.scaler = None

        # Initialize PowerTransformer
        # We set standardize=False here because we apply StandardScaler explicitly
        # in the next step, allowing for granular control via config.
        if USE_POWER_TRANSFORM:
            self.pt = PowerTransformer(method=POWER_TRANSFORM_METHOD, standardize=False)

        # Initialize StandardScaler
        if USE_STANDARD_SCALING:
            self.scaler = StandardScaler()

    def process_and_cache(self, X_train, X_val, X_test, load_cached_data=True):
        """
        Applies the preprocessing pipeline to the training, validation, and test sets.

        Logic:
        1. If load_cached_data is True and cache files exist, load and return them.
        2. Otherwise, fit transformers on X_train and transform all sets.
        3. Save the transformed sets to cache files.

        Args:
            X_train (np.ndarray): Raw training features.
            X_val (np.ndarray): Raw validation features.
            X_test (np.ndarray): Raw test features.
            load_cached_data (bool): Flag to attempt loading from cache.

        Returns:
            tuple: (X_train_transformed, X_val_transformed, X_test_transformed)
        """
        # Set seed for reproducibility (though sklearn transformers are mostly deterministic)
        set_seed(SEED)

        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # 1. Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(CACHE_TRAIN_PATH)
                and os.path.exists(CACHE_VAL_PATH)
                and os.path.exists(CACHE_TEST_PATH)
            ):

                print(f"Loading preprocessed data from cache at {WORKING_DIR}...")
                try:
                    X_train_trans = np.load(CACHE_TRAIN_PATH)
                    X_val_trans = np.load(CACHE_VAL_PATH)
                    X_test_trans = np.load(CACHE_TEST_PATH)
                    return X_train_trans, X_val_trans, X_test_trans
                except Exception as e:
                    print(f"Error loading cache: {e}. Proceeding to re-process data.")
            else:
                print("Cache files not found. Proceeding to process data.")
        else:
            print("Cache usage disabled. Proceeding to process data.")

        # 2. Process Data
        print("Starting preprocessing pipeline...")

        # Create copies to avoid modifying original arrays
        X_train_proc = X_train.copy()
        X_val_proc = X_val.copy()
        X_test_proc = X_test.copy()

        # Apply Power Transformation
        if self.pt is not None:
            print(f"Applying PowerTransformer (method='{POWER_TRANSFORM_METHOD}')...")
            # Fit on Train, Transform Train/Val/Test
            X_train_proc = self.pt.fit_transform(X_train_proc)
            X_val_proc = self.pt.transform(X_val_proc)
            X_test_proc = self.pt.transform(X_test_proc)

        # Apply Standard Scaling
        if self.scaler is not None:
            print("Applying StandardScaler...")
            # Fit on Train, Transform Train/Val/Test
            X_train_proc = self.scaler.fit_transform(X_train_proc)
            X_val_proc = self.scaler.transform(X_val_proc)
            X_test_proc = self.scaler.transform(X_test_proc)

        # 3. Save to Cache
        print(f"Saving processed data to {WORKING_DIR}...")
        try:
            np.save(CACHE_TRAIN_PATH, X_train_proc)
            np.save(CACHE_VAL_PATH, X_val_proc)
            np.save(CACHE_TEST_PATH, X_test_proc)
        except Exception as e:
            print(f"Warning: Failed to save cache files: {e}")

        return X_train_proc, X_val_proc, X_test_proc
