from library.utils import load_processed_data


def load_datasets(load_cached_data=True, debug=False, debug_size=500):
    """
    Handles the ingestion of datasets from the metadata directory.

    This function loads the training, validation, and test datasets. It utilizes
    the library utility to handle reading from CSVs, ensuring correct data types
    (specifically string formatting for text columns), performing basic preprocessing,
    and managing a caching mechanism to improve performance on subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed data from
                                 the cache directory (./working/idea_1). If False
                                 or if the cache is invalid/missing, the data is
                                 processed from scratch and cached.
        debug (bool): If True, returns a truncated version of the datasets (subset)
                      determined by `debug_size`. Useful for rapid testing.
        debug_size (int): The number of rows to include in the subset when
                          debug is True.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - train_df: Training data with columns ['textID', 'text', 'selected_text', 'sentiment', ...].
            - val_df: Validation data with columns ['textID', 'text', 'selected_text', 'sentiment', ...].
            - test_df: Test data with columns ['textID', 'text', 'sentiment', ...].
    """
    # Delegate the data loading, preprocessing, and caching logic to the provided utility
    return load_processed_data(
        load_cached_data=load_cached_data, debug=debug, debug_size=debug_size
    )
