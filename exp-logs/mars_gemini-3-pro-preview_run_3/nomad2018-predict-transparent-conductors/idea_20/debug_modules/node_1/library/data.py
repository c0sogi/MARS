import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TARGET_COLS,
    SAMPLE_SIZE,
    RANDOM_SEED,
)
from library.features import generate_features


def load_metadata(dataset_type):
    """
    Loads metadata from CSV files.
    Applies sampling if SAMPLE_SIZE is set in config.
    Resets index to ensure alignment with generated features.
    """
    if dataset_type == "train":
        path = TRAIN_METADATA_PATH
    elif dataset_type == "val":
        path = VAL_METADATA_PATH
    elif dataset_type == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # Apply sampling for debugging if configured
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    # Crucial: Reset index so that pd.concat in generate_features works correctly
    # generate_features creates a DataFrame from a list (index 0..N) and concats it
    # with metadata_df. If metadata_df index is not reset, rows won't align.
    df = df.reset_index(drop=True)

    return df


def get_dataset_name(base_name):
    """
    Constructs a cache-friendly dataset name.
    Appends sample size suffix if sampling is active to avoid cache collisions.
    """
    if SAMPLE_SIZE is not None:
        return f"{base_name}_sample_{SAMPLE_SIZE}"
    return base_name


def load_dataset(dataset_type, load_cached_data=True):
    """
    Main entry point to load X (features) and y (targets).

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached feature files.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame or None): Target matrix (log1p transformed).
    """
    # 1. Load Metadata
    metadata_df = load_metadata(dataset_type)

    # 2. Determine cache key
    cache_name = get_dataset_name(dataset_type)

    # 3. Generate or Load Features
    # This function handles the heavy lifting: geometry parsing, RDF/CTM calculation, and caching.
    # It returns a dataframe containing both the new geometric features and the original tabular features.
    features_df = generate_features(
        metadata_df, cache_name, load_cached_data=load_cached_data
    )

    # 4. Separate Features and Targets
    # The generate_features function excludes target columns from the returned dataframe
    # to keep it as a pure feature matrix. We need to grab targets from the metadata.

    # Ensure alignment: metadata_df index was reset in load_metadata,
    # and features_df preserves that order.

    if dataset_type in ["train", "val"]:
        # Extract targets
        y_raw = metadata_df[TARGET_COLS].copy()

        # Apply Log transformation: z = log(1 + y)
        # This helps with the skewed distribution of energies and ensures positivity constraints
        y = np.log1p(y_raw)

        # X is the features dataframe
        X = features_df

        return X, y
    else:
        # For test set, there are no targets
        return features_df, None


def inverse_transform_targets(y_pred_log):
    """
    Applies the inverse transformation to predictions: y = exp(z) - 1
    """
    return np.expm1(y_pred_log)
