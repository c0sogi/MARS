import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR, FLOAT_PRECISION, PIPELINE_CONFIG, SEED
from library.data_loader import get_data_loaders


class Float64Pipeline:
    """
    A wrapper around sklearn transformers to enforce float64 precision
    and implement the specific inductive preprocessing steps.
    """

    def __init__(self):
        self.yeo_johnson = PIPELINE_CONFIG.get("yeo_johnson", False)
        self.standard_scaler = PIPELINE_CONFIG.get("standard_scaler", False)
        self.pt = None
        self.ss = None

        # Initialize PowerTransformer if configured
        if self.yeo_johnson:
            standardize = PIPELINE_CONFIG.get("yeo_johnson_standardize", False)
            self.pt = PowerTransformer(method="yeo-johnson", standardize=standardize)

        # Initialize StandardScaler if configured
        if self.standard_scaler:
            self.ss = StandardScaler()

    def fit(self, X):
        """
        Fits the pipeline on the provided data (Training set).
        """
        # Enforce high precision
        X_data = np.array(X, dtype=FLOAT_PRECISION)

        curr_X = X_data

        if self.pt:
            self.pt.fit(curr_X)
            # Transform to pass to next step if needed (though fit_transform is not used here for clarity)
            curr_X = self.pt.transform(curr_X)

        if self.ss:
            self.ss.fit(curr_X)

        return self

    def transform(self, X):
        """
        Transforms the data using the fitted parameters.
        """
        # Enforce high precision
        X_data = np.array(X, dtype=FLOAT_PRECISION)

        curr_X = X_data

        if self.pt:
            curr_X = self.pt.transform(curr_X)

        if self.ss:
            curr_X = self.ss.transform(curr_X)

        return curr_X.astype(FLOAT_PRECISION)

    def fit_transform(self, X):
        """
        Convenience method for fit and transform.
        """
        self.fit(X)
        return self.transform(X)


def get_preprocessed_data(load_cached_data=True, limit=None):
    """
    Retrieves the raw data, applies the Float64Pipeline, and handles caching of the transformed features.

    Args:
        load_cached_data (bool): Whether to use cached .npy files if available.
        limit (int, optional): Limit dataset size for debugging.

    Returns:
        tuple: (train_data, val_data, test_data)
               Each is a tuple of (X_transformed, y, ids).
    """
    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "X_train_transformed.npy")
    cache_val_path = os.path.join(WORKING_DIR, "X_val_transformed.npy")
    cache_test_path = os.path.join(WORKING_DIR, "X_test_transformed.npy")

    # 1. Retrieve Raw Data (Features, Labels, IDs)
    # We always call this to get y and ids, and to get X if we need to compute.
    # data_loader handles its own caching of the raw merge.
    (
        (X_train_raw, y_train, ids_train),
        (X_val_raw, y_val, ids_val),
        (X_test_raw, _, ids_test),
    ) = get_data_loaders(load_cached_data=load_cached_data, limit=limit)

    # 2. Check Cache for Transformed Features
    if load_cached_data and limit is None:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):

            print("Loading preprocessed transformed data from cache...")
            X_train = np.load(cache_train_path)
            X_val = np.load(cache_val_path)
            X_test = np.load(cache_test_path)

            return (
                (X_train, y_train, ids_train),
                (X_val, y_val, ids_val),
                (X_test, None, ids_test),
            )

    # 3. Compute Transformations
    print("Computing preprocessed data (fitting pipeline)...")

    pipeline = Float64Pipeline()

    # Fit ONLY on training data (Inductive)
    print("Fitting pipeline on Training set...")
    pipeline.fit(X_train_raw)

    # Transform all splits
    print("Transforming Training set...")
    X_train_trans = pipeline.transform(X_train_raw)

    print("Transforming Validation set...")
    X_val_trans = pipeline.transform(X_val_raw)

    print("Transforming Test set...")
    X_test_trans = pipeline.transform(X_test_raw)

    # 4. Save to Cache
    # Only cache if we are not debugging with a limit
    if limit is None:
        print("Saving preprocessed data to cache...")
        # Ensure directory exists (redundant if config handled it, but safe)
        os.makedirs(WORKING_DIR, exist_ok=True)

        np.save(cache_train_path, X_train_trans)
        np.save(cache_val_path, X_val_trans)
        np.save(cache_test_path, X_test_trans)

    return (
        (X_train_trans, y_train, ids_train),
        (X_val_trans, y_val, ids_val),
        (X_test_trans, None, ids_test),
    )
