import os
import pandas as pd
import library.feature_engineering
from library.feature_engineering import FeatureSelector
from library.config import WORKING_DIR


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    load_cached_data: bool = True,
    subset_size: float = None,
) -> list:
    """
    Identifies the most predictive features using Recursive Feature Elimination (RFE).
    Wraps the FeatureSelector from the library to handle caching and execution.

    Args:
        X (pd.DataFrame): Training feature matrix.
        y (pd.Series): Target variable.
        load_cached_data (bool): If True, attempts to load selected features from disk
                                 to avoid re-computation.
        subset_size (float, optional): Fraction of data to use for RFE fitting.
                                       Overrides the default in config if provided.

    Returns:
        list: List of names of the selected features.
    """
    # Allow overriding the subset size for debugging or faster runs
    if subset_size is not None:
        print(f"Overriding RFE subset size to {subset_size}")
        library.feature_engineering.RFE_TRAIN_SUBSET_SIZE = subset_size

    # Instantiate the provided FeatureSelector
    selector = FeatureSelector(cache_dir=WORKING_DIR)
    cache_path = selector.selected_columns_path

    # Caching Logic: Check if we can load existing results
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading selected features from cache: {cache_path}")
            with open(cache_path, "r") as f:
                selected_features = [line.strip() for line in f if line.strip()]

            if len(selected_features) > 0:
                return selected_features
            else:
                print("Warning: Cached feature file is empty. Re-running selection.")
        except Exception as e:
            print(f"Error reading cache: {e}. Re-running selection.")

    # If cache miss or force recompute
    print("Starting feature selection process...")

    # The FeatureSelector.fit method handles:
    # 1. Subsampling the data (using RFE_TRAIN_SUBSET_SIZE)
    # 2. Running RFE with RandomForest
    # 3. Saving the result to cache_path
    selector.fit(X, y)

    return selector.selected_features
