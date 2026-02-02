import os
import random
import numpy as np
import torch
from library.config import Config
from library.preprocessing import get_preprocessed_data


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_process_data(load_cached_data=True, debug=False, debug_sample_size=100):
    """
    Orchestrates data ingestion, feature extraction, fusion, and preprocessing.

    This function acts as a wrapper around the library's preprocessing pipeline,
    configuring the global environment and handling debug constraints before
    delegating the heavy lifting to `library.preprocessing.get_preprocessed_data`.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
                                 Defaults to True.
        debug (bool): If True, runs in debug mode with a smaller dataset.
                      Defaults to False.
        debug_sample_size (int): Number of samples to use in debug mode.
                                 Defaults to 100.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Preprocessed training features (float64).
            - y_train (np.ndarray): Encoded training labels.
            - X_val (np.ndarray): Preprocessed validation features (float64).
            - y_val (np.ndarray): Encoded validation labels.
            - X_test (np.ndarray): Preprocessed test features (float64).
            - test_ids (np.ndarray): IDs for the test set.
            - classes (np.ndarray): Array of class names corresponding to labels.
    """
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Configure global settings based on arguments
    Config.DEBUG = debug
    Config.DEBUG_SAMPLE_SIZE = debug_sample_size

    # If debugging is enabled, we must force load_cached_data to False.
    # This ensures that the pipeline processes the subsampled metadata from scratch
    # rather than loading the full-size dataset from an existing cache.
    if debug:
        if load_cached_data:
            print(
                f"Debug mode enabled. Forcing load_cached_data=False to ensure subsampling to {debug_sample_size} samples."
            )
        load_cached_data = False

    # Delegate to the provided library implementation which handles:
    # 1. Metadata loading
    # 2. Geometric feature extraction (with polarity correction)
    # 3. Merging tabular and geometric features
    # 4. High-precision preprocessing (Yeo-Johnson + StandardScaler in float64)
    # 5. Caching of the processed arrays
    return get_preprocessed_data(load_cached_data=load_cached_data)
