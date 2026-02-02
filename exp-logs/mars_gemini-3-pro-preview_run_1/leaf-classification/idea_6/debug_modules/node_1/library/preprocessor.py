import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import Config


def get_pipeline():
    """
    Constructs and returns the feature engineering pipeline.

    The pipeline consists of:
    1. PowerTransformer (Yeo-Johnson): To stabilize variance and make features more Gaussian-like.
    2. StandardScaler: To ensure zero mean and unit variance for the subsequent LDA/GPC models.

    Returns:
        sklearn.pipeline.Pipeline: The constructed preprocessing pipeline.
    """
    pipeline = Pipeline(
        [
            (
                "power_transformer",
                PowerTransformer(
                    method=Config.POWER_TRANSFORM_METHOD, standardize=True
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline


def preprocess_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Applies the preprocessing pipeline to the dataset.
    Implements caching to avoid redundant computation.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        X_test (np.ndarray): Test features.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_processed, X_val_processed, X_test_processed)
    """
    # Define cache paths
    cache_train = Config.CACHE_TRAIN_PATH
    cache_val = Config.CACHE_VAL_PATH
    cache_test = Config.CACHE_TEST_PATH

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):

            print(f"Loading preprocessed data from {Config.WORKING_DIR}...")
            try:
                X_train_proc = np.load(cache_train)
                X_val_proc = np.load(cache_val)
                X_test_proc = np.load(cache_test)
                return X_train_proc, X_val_proc, X_test_proc
            except Exception as e:
                print(f"Error loading cache: {e}. Reprocessing...")
        else:
            print("Preprocessed cache miss. Processing from scratch...")
    else:
        print("Force reprocessing enabled. Processing from scratch...")

    # 2. Process data
    print("Fitting preprocessing pipeline...")
    pipeline = get_pipeline()

    # Fit only on training data
    X_train_proc = pipeline.fit_transform(X_train)

    # Transform validation and test data
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # 3. Save to cache
    print(f"Saving preprocessed data to {Config.WORKING_DIR}...")
    np.save(cache_train, X_train_proc)
    np.save(cache_val, X_val_proc)
    np.save(cache_test, X_test_proc)

    return X_train_proc, X_val_proc, X_test_proc
