import numpy as np
from library.config import SEED
from library.utils import set_seed
from library.preprocessing import get_preprocessed_data

# Ensure reproducibility at the module level
set_seed(SEED)


def load_data(load_cached_data=True, max_samples=None):
    """
    Orchestrates the loading, feature extraction, and preprocessing of the hybrid dataset.

    This function serves as the primary interface for accessing the data prepared for the
    Spectral-Spatial High-Precision OAS Discriminant model. It delegates the heavy lifting
    to the provided library modules to ensure consistency with the pre-defined pipeline.

    The pipeline steps (handled by library.preprocessing.get_preprocessed_data) include:
    1.  **Metadata Loading**: Reads stratified train/val and test metadata.
    2.  **Feature Extraction**:
        -   Loads binary images.
        -   Extracts **Spectral Features** (Elliptical Fourier Descriptors) to capture micro-morphology.
        -   Extracts **Spatial Features** (Area, Solidity, etc.) to capture macro-geometry.
        -   Retrieves pre-extracted tabular features (Margin, Shape, Texture).
    3.  **Merging & Sorting**: Merges all feature sets and enforces **Alphanumeric Column Ordering**
        for deterministic memory layout.
    4.  **High-Precision Preprocessing**:
        -   Applies **Yeo-Johnson Power Transformation** (standardize=False).
        -   Applies **StandardScaler**.
        -   Maintains strict **float64** precision throughout.
    5.  **Caching**: Caches the resulting numpy arrays to disk using configuration-hashed filenames
        to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed numpy arrays from the
                                 cache directory defined in config. If False or if the cache
                                 is invalid/missing, the pipeline runs from scratch.
                                 Defaults to True.
        max_samples (int, optional): If provided, limits the number of samples loaded from the
                                     raw CSVs. This is useful for rapid debugging.
                                     Note: Caching is bypassed when max_samples is set to
                                     prevent overwriting full-dataset caches with partial data.

    Returns:
        tuple: A tuple containing the processed datasets and auxiliary information:
            - X_train (np.ndarray): Preprocessed training feature matrix (float64).
            - y_train (np.ndarray): Encoded training target labels.
            - X_val (np.ndarray): Preprocessed validation feature matrix (float64).
            - y_val (np.ndarray): Encoded validation target labels.
            - X_test (np.ndarray): Preprocessed test feature matrix (float64).
            - test_ids (np.ndarray): Array of unique IDs for the test set images.
            - classes (np.ndarray): Array of class names corresponding to the encoded labels.
    """
    # Delegate to the library's high-precision preprocessing pipeline
    # This ensures we use the exact logic defined in library/preprocessing.py
    # including the caching mechanism and feature extraction calls.
    return get_preprocessed_data(
        load_cached_data=load_cached_data, max_samples=max_samples
    )
