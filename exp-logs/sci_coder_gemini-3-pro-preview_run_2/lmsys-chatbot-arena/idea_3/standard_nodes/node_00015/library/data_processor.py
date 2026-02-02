import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from library.config import Config


def clean_text(text_series):
    """
    Cleans text data by filling NaNs with empty strings and stripping whitespace.

    Args:
        text_series (pd.Series): The pandas series containing text data.

    Returns:
        pd.Series: The cleaned text series.
    """
    return text_series.fillna("").astype(str).str.strip()


def extract_meta_features(df):
    """
    Computes character lengths for 'prompt', 'response_a', and 'response_b'.

    Args:
        df (pd.DataFrame): The dataframe containing the text columns.

    Returns:
        np.ndarray: A 2D numpy array of shape (n_samples, 3) containing the lengths.
    """
    # Calculate lengths, handling potential NaNs by treating them as empty strings
    len_prompt = df["prompt"].fillna("").str.len()
    len_a = df["response_a"].fillna("").str.len()
    len_b = df["response_b"].fillna("").str.len()

    # Stack into a (N, 3) array
    return np.column_stack((len_prompt, len_a, len_b))


def get_scaler(train_meta_features):
    """
    Fits a StandardScaler on the training meta-features.

    Args:
        train_meta_features (np.ndarray): The meta-features from the training set.

    Returns:
        StandardScaler: The fitted scaler object.
    """
    scaler = StandardScaler()
    scaler.fit(train_meta_features)
    return scaler


def load_data(load_cached_data=True, debug=False, debug_size=None):
    """
    Loads, processes, and caches the training, validation, and test datasets.

    Implements the caching logic:
    1. If load_cached_data is True (and not in debug mode), attempt to load from Parquet.
    2. If cache missing or load_cached_data is False, load raw CSVs from metadata.
    3. Process text, extract meta-features, fit scaler on train, transform all.
    4. Save to cache (unless in debug mode).

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, subsamples the data for rapid testing.
        debug_size (int, optional): Number of samples to use in debug mode.
                                    Defaults to Config.DEBUG_SAMPLE_SIZE.

    Returns:
        tuple: (df_train, df_val, df_test) containing processed dataframes with
               added meta-feature columns ['meta_prompt_len', 'meta_a_len', 'meta_b_len'].
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "cached_train.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "cached_val.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "cached_test.parquet")

    # Determine if we should load from cache
    # We avoid loading cache in debug mode to ensure we get the requested subsample size
    # unless we specifically implemented a debug cache, but standard practice is to re-process for debug.
    should_load = load_cached_data and not debug

    cache_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if should_load and cache_exists:
        print("Loading data from cache...")
        try:
            df_train = pd.read_parquet(cache_train_path)
            df_val = pd.read_parquet(cache_val_path)
            df_test = pd.read_parquet(cache_test_path)
            return df_train, df_val, df_test
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")
            # Fall through to processing logic

    print("Loading raw data from metadata...")
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Handle Debugging
    if debug:
        sample_size = debug_size if debug_size is not None else Config.DEBUG_SAMPLE_SIZE
        print(f"Debug mode enabled. Subsampling to {sample_size} rows.")
        df_train = df_train.head(sample_size).copy()
        df_val = df_val.head(sample_size).copy()
        df_test = df_test.head(sample_size).copy()

    # 1. Clean Text
    print("Cleaning text...")
    text_cols = ["prompt", "response_a", "response_b"]
    for df in [df_train, df_val, df_test]:
        for col in text_cols:
            df[col] = clean_text(df[col])

    # 2. Extract Meta Features
    print("Extracting meta-features...")
    train_meta = extract_meta_features(df_train)
    val_meta = extract_meta_features(df_val)
    test_meta = extract_meta_features(df_test)

    # 3. Scale Meta Features
    # Fit scaler ONLY on training data to avoid leakage
    print("Scaling meta-features...")
    scaler = get_scaler(train_meta)

    train_meta_scaled = scaler.transform(train_meta)
    val_meta_scaled = scaler.transform(val_meta)
    test_meta_scaled = scaler.transform(test_meta)

    # 4. Attach to DataFrames
    meta_col_names = ["meta_prompt_len", "meta_a_len", "meta_b_len"]

    df_train[meta_col_names] = train_meta_scaled
    df_val[meta_col_names] = val_meta_scaled
    df_test[meta_col_names] = test_meta_scaled

    # 5. Cache Results
    # Do not overwrite main cache if in debug mode
    if not debug:
        print("Saving processed data to cache...")
        df_train.to_parquet(cache_train_path, index=False)
        df_val.to_parquet(cache_val_path, index=False)
        df_test.to_parquet(cache_test_path, index=False)

    return df_train, df_val, df_test
