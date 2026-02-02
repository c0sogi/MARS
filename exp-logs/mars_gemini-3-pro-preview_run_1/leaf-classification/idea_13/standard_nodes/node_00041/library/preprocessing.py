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

        # 1. Concatenate Train and Test for Transductive fitting
        # We use Train + Test to estimate the manifold (Transductive), but Val must remain unseen.
        parts = [X_train, X_test]
        X_fit = pd.concat(parts, axis=0, ignore_index=True)

        # 2. Fit PowerTransformer (Yeo-Johnson)
        # standardize=False because we want to apply StandardScaler explicitly afterwards
        print("Fitting PowerTransformer (Yeo-Johnson) on Train + Test...")
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.fit(X_fit)

        # 3. Fit Standard Scaler
        # We must transform the fit data first to get the scale for StandardScaler
        print("Fitting StandardScaler on Train + Test...")
        X_fit_pt = pt.transform(X_fit)
        ss = StandardScaler()
        ss.fit(X_fit_pt)

        # 4. Transform all datasets using the fitted transformers
        print("Applying transformations to all sets...")
        X_train_trans = ss.transform(pt.transform(X_train))
        X_test_trans = ss.transform(pt.transform(X_test))

        if has_val:
            X_val_trans = ss.transform(pt.transform(X_val))
        else:
            X_val_trans = None

        # 5. Save to cache
        print(f"Saving transformed data to {WORKING_DIR}...")
        np.save(cache_paths["X_train"], X_train_trans)
        np.save(cache_paths["X_test"], X_test_trans)
        if has_val:
            np.save(cache_paths["X_val"], X_val_trans)

        return X_train_trans, X_test_trans, X_val_trans
