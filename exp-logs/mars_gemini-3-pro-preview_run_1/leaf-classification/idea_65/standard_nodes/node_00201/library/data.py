import os
import numpy as np
from library import config, features, utils


class LeafDataLoader:
    """
    Manages data loading, integration, and caching for the leaf species classification task.

    This class serves as the primary interface for accessing the processed dataset.
    It leverages the 'Sanitized Parsimonious Geometric High-Precision OAS Discriminant' pipeline
    implemented in the library.features module, ensuring that:
    1. Robust geometric features (Golden 5) are extracted.
    2. Tabular features are merged.
    3. Pipeline sanitization (VarianceThreshold) is applied.
    4. Numerical stability (Yeo-Johnson + Float64) is enforced.
    5. Data is cached efficiently using configuration hashing.
    """

    def __init__(self):
        """
        Initializes the data loader.
        """
        pass

    def load_data(self, load_cached_data: bool = True, debug: bool = False):
        """
        Loads the preprocessed training, validation, and test datasets.

        This method orchestrates the retrieval of data, either from the disk cache
        or by computing it from scratch via the library pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed features
                                     from the cache directory defined in config.
                                     If False or cache miss, re-runs the pipeline.
            debug (bool): If True, overrides the configuration to run in debug mode,
                          processing only a small subset of the data for rapid testing.

        Returns:
            tuple: A tuple containing the following numpy arrays:
                - X_train (np.ndarray): Processed training features (float64).
                - y_train (np.ndarray): Encoded training labels.
                - X_val (np.ndarray): Processed validation features (float64).
                - y_val (np.ndarray): Encoded validation labels.
                - X_test (np.ndarray): Processed test features (float64).
                - test_ids (np.ndarray): IDs corresponding to test images.
                - classes (np.ndarray): Array of original class names.
        """
        # Ensure reproducibility across all operations
        utils.set_seed(config.SEED)

        # Manage Debug State
        # We temporarily override the global config setting if the method argument dictates it.
        # This allows the underlying library functions to respect the runtime flag without
        # requiring changes to the library code itself.
        original_debug_state = config.DEBUG
        if debug:
            config.DEBUG = True

        try:
            # Delegate to the library's robust data retrieval function.
            # This function strictly implements the required pipeline:
            # - Metadata loading from ./metadata
            # - Feature extraction (Geometric + Tabular)
            # - Pipeline Sanitization (VarianceThreshold)
            # - Transformation (Yeo-Johnson) & Scaling
            # - Caching logic (Hash generation, Save/Load to ./working/idea_65)
            return features.get_data(load_cached_data=load_cached_data)

        except Exception as e:
            # Ensure errors are propagated clearly
            print(f"An error occurred during data loading: {e}")
            raise e

        finally:
            # Always restore the original configuration state
            config.DEBUG = original_debug_state
