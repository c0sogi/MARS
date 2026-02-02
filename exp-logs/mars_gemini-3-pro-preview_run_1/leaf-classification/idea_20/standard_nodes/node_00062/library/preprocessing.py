import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import WORKING_DIR


def create_preprocessor():
    """
    Creates the feature transformation pipeline.

    Strategy:
    1. PowerTransformer (Yeo-Johnson): Stabilizes variance and makes data more Gaussian-like.
       standardize=False is used because we apply a dedicated StandardScaler next.
    2. StandardScaler: Centers and scales the data to unit variance.

    Returns:
        sklearn.pipeline.Pipeline: The configured preprocessing pipeline.
    """
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


def get_transformed_data(X_train, X_val, X_test, load_cached_data=True):
    """
    Applies the preprocessing pipeline inductively (fitting only on train) and handles caching.

    Strictly enforces float64 precision by maintaining the input dtype.

    Args:
        X_train (np.ndarray): Training features (float64).
        X_val (np.ndarray): Validation features (float64).
        X_test (np.ndarray): Test features (float64).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_transformed, X_val_transformed, X_test_transformed)
    """
    # Define cache paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train_transformed.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val_transformed.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test_transformed.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading transformed features from {WORKING_DIR}...")
            X_train_trans = np.load(cache_files["X_train"])
            X_val_trans = np.load(cache_files["X_val"])
            X_test_trans = np.load(cache_files["X_test"])
            print("Transformed data loaded successfully.")
            return X_train_trans, X_val_trans, X_test_trans
        else:
            print(
                "Transformed cache not found or incomplete. Processing from scratch..."
            )

    # Create pipeline
    pipeline = create_preprocessor()

    print("Fitting preprocessor on Training set (Inductive Fit)...")
    # Fit ONLY on training data to avoid data leakage
    pipeline.fit(X_train)

    print("Transforming datasets...")
    # Transform all sets using the parameters derived from train
    X_train_trans = pipeline.transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    # Verify precision (sanity check)
    if X_train_trans.dtype != np.float64:
        print("Warning: Transformed data is not float64. Casting back to float64.")
        X_train_trans = X_train_trans.astype(np.float64)
        X_val_trans = X_val_trans.astype(np.float64)
        X_test_trans = X_test_trans.astype(np.float64)

    # Save to cache
    print(f"Saving transformed features to {WORKING_DIR}...")
    np.save(cache_files["X_train"], X_train_trans)
    np.save(cache_files["X_val"], X_val_trans)
    np.save(cache_files["X_test"], X_test_trans)

    return X_train_trans, X_val_trans, X_test_trans
