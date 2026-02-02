import os
import numpy as np
from library.utils import load_dataset
from library.config import WORKING_DIR


def load_datasets(load_cached=True):
    """
    Loads the train, validation, and test datasets separately.

    Args:
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        tuple: ((X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test))
    """
    print("Loading train, validation, and test datasets...")

    # Load Train
    X_train, y_train, ids_train = load_dataset("train", load_cached=load_cached)

    # Load Validation
    X_val, y_val, ids_val = load_dataset("val", load_cached=load_cached)

    # Load Test (y is None)
    X_test, _, ids_test = load_dataset("test", load_cached=load_cached)

    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)


def get_combined_train_data(load_cached=True):
    """
    Merges the training and validation datasets into a single training set
    and returns it along with the test set. This is used for the final
    model training to maximize data utilization.

    Args:
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_combined, y_combined, X_test, ids_test)
            X_combined (np.ndarray): Merged feature matrix.
            y_combined (np.ndarray): Merged target labels.
            X_test (np.ndarray): Test feature matrix.
            ids_test (np.ndarray): Test image IDs.
    """
    # Ensure cache directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths for combined data
    x_comb_path = os.path.join(WORKING_DIR, "X_train_combined.npy")
    y_comb_path = os.path.join(WORKING_DIR, "y_train_combined.npy")

    # Always load test data first as it is required for the return
    X_test, _, ids_test = load_dataset("test", load_cached=load_cached)

    # Check if combined data is cached
    if load_cached and os.path.exists(x_comb_path) and os.path.exists(y_comb_path):
        print("Loading combined training data from cache...")
        X_combined = np.load(x_comb_path)
        y_combined = np.load(y_comb_path, allow_pickle=True)
        return X_combined, y_combined, X_test, ids_test

    print("Generating combined training data from splits...")

    # Load individual splits
    # We force load_cached=load_cached here to respect the user's wish for the underlying data
    (X_train, y_train, _), (X_val, y_val, _), _ = load_datasets(load_cached=load_cached)

    # Concatenate features and labels
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))

    # Save to cache
    print(f"Saving combined data to {WORKING_DIR}...")
    np.save(x_comb_path, X_combined)
    np.save(y_comb_path, y_combined)

    return X_combined, y_combined, X_test, ids_test
