import numpy as np
import scipy.sparse as sp
from library.features import FeatureEngineeringPipeline
from library.utils import setup_logger


def load_and_process_data(
    load_cached_data: bool = True, debug: bool = False, debug_size: int = 100
):
    """
    Orchestrates data ingestion and preparation by invoking the FeatureEngineeringPipeline.

    This function handles the loading of raw data (via the pipeline), generation of
    Lexical, Behavioral, Semantic, and Metadata feature views, and application of
    preprocessing steps like imputation and scaling.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed features from the cache directory.
                                 If False or cache is missing, re-runs the full feature engineering pipeline.
        debug (bool): If True, subsets the resulting data dictionaries to `debug_size` samples.
                      Useful for quick testing of training loops.
        debug_size (int): The number of samples to retain in each split if debug is True.

    Returns:
        dict: A nested dictionary containing processed data for 'train', 'val', and 'test' splits.
              Structure:
              {
                  "train": {
                      "lexical": sparse_matrix,
                      "behavioral": sparse_matrix,
                      "semantic": numpy_array,
                      "metadata": numpy_array,
                      "y": numpy_array
                  },
                  "val": { ... },
                  "test": { ... }
              }
    """
    logger = setup_logger("DataLoader")
    logger.info(f"Starting data loading process. Cache enabled: {load_cached_data}")

    # Initialize and run the provided feature engineering pipeline
    # The pipeline handles reading Parquet files, generating features, imputation, scaling, and caching.
    pipeline = FeatureEngineeringPipeline(load_cached_data=load_cached_data)
    data = pipeline.run()

    # Handle debug slicing if requested
    if debug:
        logger.info(
            f"Debug mode enabled. Slicing datasets to first {debug_size} samples."
        )
        for split in ["train", "val", "test"]:
            if split not in data:
                continue

            split_data = data[split]

            # Determine the number of samples available in this split
            # We use 'metadata' as the reference for the number of rows
            if "metadata" in split_data:
                n_samples = split_data["metadata"].shape[0]
            else:
                # Fallback to any available key
                first_key = next(iter(split_data))
                n_samples = split_data[first_key].shape[0]

            limit = min(n_samples, debug_size)

            # Slice all arrays/matrices in the split dictionary to ensure alignment
            for key, value in split_data.items():
                if sp.issparse(value):
                    # Slice sparse matrix (CSR/CSC)
                    split_data[key] = value[:limit]
                elif isinstance(value, np.ndarray):
                    # Slice numpy array (dense features or targets)
                    split_data[key] = value[:limit]
                elif isinstance(value, list):
                    # Slice lists if any
                    split_data[key] = value[:limit]

            data[split] = split_data
            logger.info(f"Sliced '{split}' set to {limit} samples.")

    logger.info("Data loading and processing complete.")
    return data
