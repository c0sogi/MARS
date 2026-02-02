import os
import pandas as pd
import library.config as config
import library.utils as utils


def preprocess_dataset(df):
    """
    Applies text preprocessing to the 'text' column of the DataFrame.
    Performs lowercasing if enabled in the configuration.

    Args:
        df (pd.DataFrame): The dataframe containing a 'text' column.

    Returns:
        pd.DataFrame: The dataframe with processed text.
    """
    # Create a copy to avoid SettingWithCopy warnings if a slice is passed
    df = df.copy()

    if "text" in df.columns and config.LOWERCASE:
        df["text"] = df["text"].str.lower()

    return df


def load_and_preprocess_data(load_cached_data=True, nrows=None):
    """
    Loads the training, validation, and test datasets.
    Implements a caching mechanism to save processed data to disk.

    Args:
        load_cached_data (bool): If True, attempts to load data from the cache directory.
        nrows (int, optional): Number of rows to read. Useful for debugging.

    Returns:
        tuple: A tuple containing (train_df, val_df, test_df).
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    train_cache_path = os.path.join(config.CACHE_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(config.CACHE_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(config.CACHE_DIR, "test_processed.parquet")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data:
        try:
            if (
                os.path.exists(train_cache_path)
                and os.path.exists(val_cache_path)
                and os.path.exists(test_cache_path)
            ):

                print("Loading preprocessed data from cache...")
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)

                # If debugging with a subset, slice the cached data
                if nrows is not None:
                    train_df = train_df.head(nrows)
                    val_df = val_df.head(nrows)
                    test_df = test_df.head(nrows)

                return train_df, val_df, test_df
            else:
                print("Cache files not found. Processing from scratch...")
        except Exception as e:
            print(f"Error loading cache: {e}. Processing from scratch...")

    # 2. IF loading fails OR load_cached_data is False:
    print("Loading raw datasets...")
    train_df = utils.load_dataset(config.TRAIN_PATH, nrows=nrows)
    val_df = utils.load_dataset(config.VAL_PATH, nrows=nrows)
    test_df = utils.load_dataset(config.TEST_PATH, nrows=nrows)

    print("Preprocessing text data...")
    train_df = preprocess_dataset(train_df)
    val_df = preprocess_dataset(val_df)
    test_df = preprocess_dataset(test_df)

    # Save to cache only if we processed the full dataset (nrows is None)
    # This ensures we don't save a partial debug dataset as the cache
    if nrows is None:
        print(f"Saving processed data to cache at {config.CACHE_DIR}...")
        try:
            train_df.to_parquet(train_cache_path, index=False)
            val_df.to_parquet(val_cache_path, index=False)
            test_df.to_parquet(test_cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    return train_df, val_df, test_df
