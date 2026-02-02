import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import load_or_compute
from library.mace_embedding import extract_features


def build_feature_matrix(
    split: str, load_cached_data: bool = True, limit: int = None
) -> pd.DataFrame:
    """
    Constructs the complete feature matrix for a given data split (train, val, test).

    This function orchestrates the data assembly by calling the MACE feature extractor,
    which combines:
    1. Tabular metadata (composition, lattice parameters)
    2. Physical descriptors (Volume, Density)
    3. Aggregated MACE structural embeddings

    The result is cached to disk to prevent redundant computation.

    Args:
        split (str): The dataset split to load ('train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load the processed dataframe from
                                 the local cache before computing it.
        limit (int, optional): If provided, limits the returned dataframe to the first N rows.
                               Useful for debugging and quick iterations.

    Returns:
        pd.DataFrame: The processed feature matrix including target columns (for train/val)
                      and the 'id' column.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define the cache file path for this specific split
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_processed.parquet")

    def _compute_full_dataset():
        # Delegate the heavy lifting to the mace_embedding module which handles
        # sub-component caching (descriptors, embeddings) and merging.
        return extract_features(split, load_cached_data=load_cached_data)

    # Load from cache or compute from scratch
    df = load_or_compute(
        cache_path=cache_path,
        compute_func=_compute_full_dataset,
        load_cached_data=load_cached_data,
    )

    # Apply dataset limiting if requested (post-loading/computation)
    if limit is not None:
        df = df.iloc[:limit].copy()

    return df


def transform_targets(targets):
    """
    Applies the logarithmic transformation to target variables.
    Transformation: z = log(1 + y)

    This aligns the regression objective with the RMSLE metric.

    Args:
        targets (np.ndarray or pd.DataFrame): The raw target values (energy).

    Returns:
        np.ndarray or pd.DataFrame: The log-transformed targets.
    """
    # Ensure no negative values are passed to log (clipping at 0)
    # Physical energies like bandgap should be non-negative.
    return np.log1p(np.maximum(targets, 0))


def inverse_transform_targets(transformed_targets):
    """
    Applies the inverse logarithmic transformation to predictions.
    Transformation: y = exp(z) - 1

    Args:
        transformed_targets (np.ndarray or pd.DataFrame): The log-scale predictions.

    Returns:
        np.ndarray or pd.DataFrame: The predictions in the original energy scale.
    """
    # Invert the log1p operation
    preds = np.expm1(transformed_targets)
    # Enforce non-negativity constraint on physical energies
    return np.maximum(preds, 0)
