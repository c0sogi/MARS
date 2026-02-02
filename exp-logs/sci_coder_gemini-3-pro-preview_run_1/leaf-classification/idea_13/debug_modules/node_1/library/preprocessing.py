import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR


class TransductivePreprocessor:
    """
    Implements the Transductive Gaussian transformation strategy.
    Leverages the combined distribution of training, validation, and test sets
    to estimate transformation parameters for maximum normality.
    """

    def __init__(self):
        pass

    def process_and_cache(self, X_train, X_test, X_val=None, load_cached_data=True):
        """
        Applies Transductive Yeo-Johnson transformation + Standardization.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            X_val (pd.DataFrame, optional): Validation features.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            tuple: (X_train_trans, X_test_trans, X_val_trans) as numpy arrays.
                   If X_val is None, X_val_trans will be None.
        """

        # Define cache paths
        cache_paths = {
            "X_train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
            "X_test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
            "X_val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        }

        # Determine if we need to process X_val
        has_val = X_val is not None

        # Check cache existence
        cache_exists = os.path.exists(cache_paths["X_train"]) and os.path.exists(
            cache_paths["X_test"]
        )
        if has_val:
            cache_exists = cache_exists and os.path.exists(cache_paths["X_val"])

        # Load from cache if requested and valid
        if load_cached_data and cache_exists:
            print("Loading transformed data from cache...")
            try:
                X_train_trans = np.load(cache_paths["X_train"])
                X_test_trans = np.load(cache_paths["X_test"])
                X_val_trans = np.load(cache_paths["X_val"]) if has_val else None
                return X_train_trans, X_test_trans, X_val_trans
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print("Computing Transductive Gaussian Transformations...")

        # 1. Concatenate all available data (Transductive step)
        # We use the union of all samples to estimate the manifold
        parts = [X_train, X_test]
        if has_val:
            parts.append(X_val)

        # Keep track of indices to split back later
        train_len = len(X_train)
        test_len = len(X_test)

        X_all = pd.concat(parts, axis=0, ignore_index=True)

        # 2. Apply Yeo-Johnson Power Transformer
        # standardize=False because we want to apply StandardScaler explicitly afterwards
        # to ensure perfect centering/scaling for the LDA solver.
        print("Fitting PowerTransformer (Yeo-Johnson)...")
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        X_all_pt = pt.fit_transform(X_all)

        # 3. Apply Standard Scaler
        print("Fitting StandardScaler...")
        ss = StandardScaler()
        X_all_scaled = ss.fit_transform(X_all_pt)

        # 4. Split back into components
        X_train_trans = X_all_scaled[:train_len]
        X_test_trans = X_all_scaled[train_len : train_len + test_len]

        if has_val:
            X_val_trans = X_all_scaled[train_len + test_len :]
        else:
            X_val_trans = None

        # 5. Save to cache
        print(f"Saving transformed data to {WORKING_DIR}...")
        np.save(cache_paths["X_train"], X_train_trans)
        np.save(cache_paths["X_test"], X_test_trans)
        if has_val:
            np.save(cache_paths["X_val"], X_val_trans)

        return X_train_trans, X_test_trans, X_val_trans
