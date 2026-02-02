import os
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
import library.config as config


def get_preprocessor():
    """
    Constructs the preprocessing pipeline.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline with PowerTransformer and StandardScaler.
    """
    # As per idea description: Yeo-Johnson Power Transformation (standardize=False)
    # followed by Standard Scaling.
    pipeline = Pipeline(
        [
            (
                "power_transformer",
                PowerTransformer(method="yeo-johnson", standardize=False),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline


def preprocess_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Applies the preprocessing pipeline to the data.

    Fits the pipeline on X_train only, then transforms X_train, X_val, and X_test.
    Enforces float64 precision and implements caching.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as numpy arrays.
    """
    # Define cache file paths
    cache_X_train_trans = os.path.join(config.CACHE_DIR, "X_train_transformed.npy")
    cache_X_val_trans = os.path.join(config.CACHE_DIR, "X_val_transformed.npy")
    cache_X_test_trans = os.path.join(config.CACHE_DIR, "X_test_transformed.npy")

    # Check if cache exists
    files_exist = all(
        os.path.exists(f)
        for f in [cache_X_train_trans, cache_X_val_trans, cache_X_test_trans]
    )

    if load_cached_data and files_exist:
        print("Loading transformed data from cache...")
        X_train_trans = np.load(cache_X_train_trans)
        X_val_trans = np.load(cache_X_val_trans)
        X_test_trans = np.load(cache_X_test_trans)
        return X_train_trans, X_val_trans, X_test_trans

    print("Fitting and applying preprocessing pipeline...")

    # Ensure input is float64 (though data_loader should have handled this, we double check/enforce)
    # Sklearn transformers usually return float64 by default if input is float64.

    pipeline = get_preprocessor()

    # Inductive Fit: Fit ONLY on training data
    pipeline.fit(X_train)

    # Transform all sets
    X_train_trans = pipeline.transform(X_train).astype(config.DTYPE)
    X_val_trans = pipeline.transform(X_val).astype(config.DTYPE)
    X_test_trans = pipeline.transform(X_test).astype(config.DTYPE)

    # Save to cache
    print(f"Saving transformed data to {config.CACHE_DIR}...")
    np.save(cache_X_train_trans, X_train_trans)
    np.save(cache_X_val_trans, X_val_trans)
    np.save(cache_X_test_trans, X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
