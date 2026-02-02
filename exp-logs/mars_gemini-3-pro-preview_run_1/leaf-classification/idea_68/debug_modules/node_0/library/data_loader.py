import pandas as pd
import library.config as config
import library.image_processing as ip
import library.utils as utils


def load_and_fuse_data(metadata_path, cache_path, load_cached_data=True, limit=None):
    """
    Loads metadata, fuses tabular and geometric features, and returns a high-precision DataFrame.

    This function implements the 'Dictionary-Based Assembly' by delegating to the
    image_processing module which computes features into a dictionary and merges them.
    It ensures that the resulting DataFrame contains all features defined in
    config.ALL_FEATURES (which enforces Alphanumeric Column Ordering).

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path to the parquet cache file.
        load_cached_data (bool): Whether to attempt loading from cache.
        limit (int, optional): If set, truncates the dataset to the first 'limit' rows.
                               Useful for debugging or quick iterations.

    Returns:
        pd.DataFrame: The processed dataframe with fused features in float64 precision.
    """
    # Ensure deterministic behavior
    utils.set_seed(config.SEED)

    # Delegate the heavy lifting (extraction, merging, caching) to the image_processing library
    # as per the requirement to use provided functions.
    df = ip.process_dataset(metadata_path, cache_path, load_cached_data)

    # Verify that the 'Alphanumeric Column Ordering' is respected by checking
    # that all configured features are present. The model will select these
    # columns using config.ALL_FEATURES to ensure the exact memory layout.
    missing_features = [f for f in config.ALL_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Data loading failed. Missing features: {missing_features}")

    # Apply debugging limit if requested
    if limit is not None:
        if limit < len(df):
            df = df.iloc[:limit].copy()

    return df


def load_datasets(load_cached_data=True, limit=None):
    """
    Loads the Training, Validation, and Test datasets using the configured paths.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.
        limit (int, optional): Limit the number of samples for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Load Training Data
    train_df = load_and_fuse_data(
        metadata_path=config.TRAIN_METADATA_PATH,
        cache_path=config.CACHE_TRAIN_PATH,
        load_cached_data=load_cached_data,
        limit=limit,
    )

    # Load Validation Data
    val_df = load_and_fuse_data(
        metadata_path=config.VAL_METADATA_PATH,
        cache_path=config.CACHE_VAL_PATH,
        load_cached_data=load_cached_data,
        limit=limit,
    )

    # Load Test Data
    test_df = load_and_fuse_data(
        metadata_path=config.TEST_METADATA_PATH,
        cache_path=config.CACHE_TEST_PATH,
        load_cached_data=load_cached_data,
        limit=limit,
    )

    return train_df, val_df, test_df
