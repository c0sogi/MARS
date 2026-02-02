import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from library.config import PCA_VARIANCE, RANDOM_SEED, WORKING_DIR


def get_scaler():
    """
    Factory function to return a configured StandardScaler.

    Returns:
        StandardScaler: The scaler object.
    """
    return StandardScaler()


def get_pca():
    """
    Factory function to return a configured PCA object.

    Returns:
        PCA: The PCA object configured with the variance threshold from config.
    """
    return PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)


def preprocess_data(
    X_train,
    X_val=None,
    X_test=None,
    use_pca=False,
    cache_prefix="data",
    load_cached=True,
):
    """
    Applies feature scaling and optional PCA transformation to the data.
    Strictly fits the transformations ONLY on X_train to prevent data leakage.
    Implements caching to disk to speed up subsequent runs.

    Args:
        X_train (np.ndarray): Training features (used for fitting).
        X_val (np.ndarray, optional): Validation features.
        X_test (np.ndarray, optional): Test features.
        use_pca (bool): Whether to apply PCA after scaling.
        cache_prefix (str): Prefix for cache files to distinguish splits/folds.
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans)
               Returns None for X_val/X_test if the input was None.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define suffix based on configuration
    suffix = "pca" if use_pca else "scaled"

    # Define cache file paths
    train_path = os.path.join(WORKING_DIR, f"{cache_prefix}_X_train_{suffix}.npy")
    val_path = os.path.join(WORKING_DIR, f"{cache_prefix}_X_val_{suffix}.npy")
    test_path = os.path.join(WORKING_DIR, f"{cache_prefix}_X_test_{suffix}.npy")

    # Check if we can load from cache
    # We only check for files corresponding to inputs that are not None
    files_exist = os.path.exists(train_path)
    if X_val is not None:
        files_exist = files_exist and os.path.exists(val_path)
    if X_test is not None:
        files_exist = files_exist and os.path.exists(test_path)

    if load_cached and files_exist:
        print(
            f"Loading preprocessed data ({suffix}) from cache with prefix '{cache_prefix}'..."
        )
        X_train_out = np.load(train_path)
        X_val_out = np.load(val_path) if X_val is not None else None
        X_test_out = np.load(test_path) if X_test is not None else None
        return X_train_out, X_val_out, X_test_out

    print(f"Preprocessing data ({suffix}) with prefix '{cache_prefix}'...")

    # 1. Scaling (Always applied)
    scaler = get_scaler()
    # Fit ONLY on training data
    X_train_out = scaler.fit_transform(X_train)

    # Transform others if they exist
    X_val_out = scaler.transform(X_val) if X_val is not None else None
    X_test_out = scaler.transform(X_test) if X_test is not None else None

    # 2. PCA (Optional)
    if use_pca:
        pca = get_pca()
        # Fit ONLY on scaled training data
        X_train_out = pca.fit_transform(X_train_out)

        # Transform others if they exist
        if X_val_out is not None:
            X_val_out = pca.transform(X_val_out)
        if X_test_out is not None:
            X_test_out = pca.transform(X_test_out)

    # 3. Save to cache
    print(f"Saving preprocessed data to {WORKING_DIR}...")
    np.save(train_path, X_train_out)

    if X_val_out is not None:
        np.save(val_path, X_val_out)

    if X_test_out is not None:
        np.save(test_path, X_test_out)

    return X_train_out, X_val_out, X_test_out
