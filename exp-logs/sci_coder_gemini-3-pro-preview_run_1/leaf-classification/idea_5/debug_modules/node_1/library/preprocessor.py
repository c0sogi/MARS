import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import Config


class FeaturePipeline:
    """
    A wrapper class for the feature transformation pipeline.
    Sequentially applies PowerTransformer (Yeo-Johnson) and StandardScaler
    to enforce multivariate Gaussian distribution assumptions required by LDA.
    """

    def __init__(self):
        # Retrieve configuration
        self.method = Config.PREPROCESSING.get("method", "yeo-johnson")
        self.do_standardize = Config.PREPROCESSING.get("standardize", True)

        # Initialize transformers
        # We set standardize=False in PowerTransformer to explicitly decouple
        # the power transform from the z-score scaling, as per the architecture design.
        self.pt = PowerTransformer(method=self.method, standardize=False)
        self.scaler = StandardScaler() if self.do_standardize else None

    def fit(self, X, y=None):
        """
        Fits the transformers on the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray, optional): Target labels (unused, for compatibility).

        Returns:
            self: The fitted pipeline.
        """
        # Fit PowerTransformer to learn lambda parameters
        self.pt.fit(X)

        # Transform data to intermediate state to fit the Scaler
        if self.scaler:
            X_pt = self.pt.transform(X)
            self.scaler.fit(X_pt)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to the data.

        Args:
            X (np.ndarray): Features to transform.

        Returns:
            np.ndarray: Transformed features.
        """
        # Apply Power Transform
        X_trans = self.pt.transform(X)

        # Apply Standard Scaling
        if self.scaler:
            X_trans = self.scaler.transform(X_trans)

        return X_trans

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data in one step.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray, optional): Target labels.

        Returns:
            np.ndarray: Transformed features.
        """
        self.fit(X, y)
        return self.transform(X)


def preprocess_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Applies the feature pipeline to the datasets with caching support.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        X_test (np.ndarray): Test features.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans)
    """

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train_preprocessed.npy"),
        "X_val": os.path.join(Config.CACHE_DIR, "X_val_preprocessed.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test_preprocessed.npy"),
    }

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_files.values()):
            print("Loading preprocessed data from cache...")
            try:
                X_train_trans = np.load(cache_files["X_train"])
                X_val_trans = np.load(cache_files["X_val"])
                X_test_trans = np.load(cache_files["X_test"])
                return X_train_trans, X_val_trans, X_test_trans
            except Exception as e:
                print(f"Error loading preprocessed cache: {e}. Recomputing...")
        else:
            print("Preprocessed cache miss. Computing...")

    # Compute transformations
    print("Fitting FeaturePipeline on training data...")
    pipeline = FeaturePipeline()

    # Fit only on training data
    pipeline.fit(X_train)

    print("Transforming datasets...")
    X_train_trans = pipeline.transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    # Save to cache
    print(f"Saving preprocessed data to {Config.CACHE_DIR}...")
    np.save(cache_files["X_train"], X_train_trans)
    np.save(cache_files["X_val"], X_val_trans)
    np.save(cache_files["X_test"], X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
