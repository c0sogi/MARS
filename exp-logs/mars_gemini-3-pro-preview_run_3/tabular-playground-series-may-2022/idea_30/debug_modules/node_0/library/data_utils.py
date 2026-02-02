import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import library.config as config


def engineer_features(df):
    """
    Performs feature engineering:
    1. Decomposes f_27 string into 10 character columns.
    2. Computes unique character count for f_27.
    """
    # 1. Unique Character Count
    # Using list comprehension for speed over .apply
    s_values = df[config.STRING_FEATURE].values.astype(str)
    unique_counts = [len(set(s)) for s in s_values]
    df[config.FEATURE_UNIQUE_COUNT] = unique_counts

    # 2. Decompose f_27 into fixed-position characters
    # f_27 is known to be length 10
    for i in range(config.F27_SEQ_LEN):
        df[f"{config.STRING_FEATURE}_{i}"] = df[config.STRING_FEATURE].str[i]

    return df


def get_all_categorical_cols():
    """
    Returns the list of all categorical feature names, including
    original categorical features and decomposed string features.
    """
    # Start with base categorical features (f_29, f_30)
    cols = list(config.CATEGORICAL_FEATURES)
    # Append decomposed f_27 features (f_27_0 ... f_27_9)
    cols += [f"{config.STRING_FEATURE}_{i}" for i in range(config.F27_SEQ_LEN)]
    return cols


def preprocess_data(train_df, val_df, test_df):
    """
    Applies transductive ordinal encoding and standard scaling.
    """
    cont_cols = config.ALL_CONTINUOUS_FEATURES
    cat_cols = get_all_categorical_cols()

    # --- 1. Transductive Ordinal Encoding ---
    # Fit on concatenation of Train + Val + Test to ensure global vocabulary alignment
    full_cat_data = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(
        dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=-1
    )
    encoder.fit(full_cat_data)

    # Transform all splits
    train_df[cat_cols] = encoder.transform(train_df[cat_cols])
    val_df[cat_cols] = encoder.transform(val_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # Calculate vocabulary sizes for embedding layers
    # encoder.categories_ is a list of arrays containing unique values for each feature
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # --- 2. Standard Scaling ---
    # Fit scaler ONLY on Training data
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    # Transform all splits (cast to float32 for memory efficiency)
    train_df[cont_cols] = scaler.transform(train_df[cont_cols]).astype(np.float32)
    val_df[cont_cols] = scaler.transform(val_df[cont_cols]).astype(np.float32)
    test_df[cont_cols] = scaler.transform(test_df[cont_cols]).astype(np.float32)

    # --- 3. Cleanup ---
    # Drop the original string column and source_path metadata
    cols_to_drop = [config.STRING_FEATURE, "source_path"]
    train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
    val_df = val_df.drop(columns=cols_to_drop, errors="ignore")
    test_df = test_df.drop(columns=cols_to_drop, errors="ignore")

    return train_df, val_df, test_df, vocab_sizes


def load_data(load_cached_data=True, nrows=None):
    """
    Loads data, performs engineering and preprocessing, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        nrows (int, optional): If set, returns only the first nrows for debugging.

    Returns:
        train_df, val_df, test_df, vocab_sizes
    """
    # Define cache paths
    train_cache = os.path.join(config.CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(config.CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(config.CACHE_DIR, "test_processed.parquet")
    vocab_cache = os.path.join(config.CACHE_DIR, "vocab_sizes.npy")

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(vocab_cache)
    ):

        print(f"Loading cached data from {config.CACHE_DIR}...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        vocab_sizes = np.load(vocab_cache).tolist()

    else:
        print("Cache not found or disabled. Processing data from scratch...")

        # Load raw data using metadata paths
        # Note: We load full datasets first to ensure correct transductive encoding
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.VAL_PATH)
        test_df = pd.read_csv(config.TEST_PATH)

        # Feature Engineering
        print("Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Preprocessing (Encoding & Scaling)
        print("Preprocessing data...")
        train_df, val_df, test_df, vocab_sizes = preprocess_data(
            train_df, val_df, test_df
        )

        # Save to cache
        print(f"Saving processed data to {config.CACHE_DIR}...")
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)
        np.save(vocab_cache, np.array(vocab_sizes))

    # Handle debugging subsample
    if nrows is not None:
        print(f"Subsampling data to {nrows} rows for debugging...")
        train_df = train_df.iloc[:nrows]
        val_df = val_df.iloc[:nrows]
        test_df = test_df.iloc[:nrows]

    return train_df, val_df, test_df, vocab_sizes
