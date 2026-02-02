import sys
import os
import numpy as np

# Add the parent directory to sys.path to allow imports from library if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library.utils import preprocess_data, set_seed


def load_data(load_cached_data=True, debug=False, max_samples=100):
    """
    Loads the Leaf Classification dataset with Global Gaussianization.

    This function utilizes the pre-implemented preprocessing pipeline in library.utils
    which applies PowerTransformer (Yeo-Johnson) to the 192 extracted features.
    It adheres to the caching mechanism and split requirements defined in the task.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from
                                 ./working/idea_20/ cache. If False or missing,
                                 reprocesses from scratch.
        debug (bool): If True, subsets the data to `max_samples` for rapid debugging.
        max_samples (int): The maximum number of samples to return when debug is True.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Gaussianized training features (n_train, 192).
            - y_train (np.ndarray): Encoded training labels (n_train,).
            - X_val (np.ndarray): Gaussianized validation features (n_val, 192).
            - y_val (np.ndarray): Encoded validation labels (n_val,).
            - X_test (np.ndarray): Gaussianized test features (n_test, 192).
            - test_ids (np.ndarray): Array of test image IDs.
            - classes (np.ndarray): Array of original class names.
    """
    # Set seed for reproducibility
    set_seed(42)

    # Load and preprocess data using the utility function
    # This handles:
    # 1. Loading metadata from ./metadata
    # 2. Extracting margin, shape, texture features
    # 3. Fitting PowerTransformer on Train, transforming Val/Test
    # 4. Caching results to ./working/idea_20
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = preprocess_data(
        load_cached_data=load_cached_data
    )

    # Handle debug mode by subsetting the data
    if debug:
        print(f"Debug mode enabled: Limiting data to {max_samples} samples.")

        # Slice training data
        if len(X_train) > max_samples:
            X_train = X_train[:max_samples]
            y_train = y_train[:max_samples]

        # Slice validation data
        if len(X_val) > max_samples:
            X_val = X_val[:max_samples]
            y_val = y_val[:max_samples]

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
