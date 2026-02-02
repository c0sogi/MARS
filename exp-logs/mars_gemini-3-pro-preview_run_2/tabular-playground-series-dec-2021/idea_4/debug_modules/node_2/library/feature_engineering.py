import os
import numpy as np
import pandas as pd
from library.config import Config


def add_interaction_features(df):
    """
    Generates multiplicative interactions between binary Wilderness_Area columns
    and continuous features.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with added interaction columns.
    """
    if not Config.ADD_INTERACTIONS:
        return df

    print("Adding interaction features...")

    # Dictionary to collect new columns before concatenation to reduce fragmentation
    new_cols = {}

    for wild_col in Config.WILDERNESS_COLS:
        for num_col in Config.NUMERIC_COLS:
            # Construct interaction feature name
            inter_col_name = f"{wild_col}_x_{num_col}"

            # Calculate interaction: Binary * Continuous
            # This preserves the continuous value where the wilderness area is active, else 0
            new_cols[inter_col_name] = df[wild_col] * df[num_col]

    # Create a DataFrame from the new columns and concatenate
    df_interactions = pd.DataFrame(new_cols, index=df.index)
    df_out = pd.concat([df, df_interactions], axis=1)

    return df_out


def get_class_map():
    """
    Returns a dictionary mapping original class labels to 0-indexed integers.
    """
    return {label: idx for idx, label in enumerate(Config.ORIGINAL_LABELS)}


def process_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        X_train (pd.DataFrame): Processed training features.
        y_train (np.ndarray): Encoded training targets.
        X_test (pd.DataFrame): Processed test features.
        test_ids (np.ndarray): Test IDs.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache files exist
    cache_exists = (
        os.path.exists(Config.CACHE_TRAIN_X)
        and os.path.exists(Config.CACHE_TRAIN_Y)
        and os.path.exists(Config.CACHE_TEST_X)
        and os.path.exists(Config.CACHE_TEST_IDS)
    )

    # Load from cache if requested and available
    if load_cached_data and cache_exists:
        print(f"Loading processed data from cache: {Config.WORKING_DIR}")
        X_train = pd.read_parquet(Config.CACHE_TRAIN_X)
        y_train = np.load(Config.CACHE_TRAIN_Y)
        X_test = pd.read_parquet(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS)
        return X_train, y_train, X_test, test_ids

    print("Cache not found or reload requested. Processing data from scratch...")

    # Load raw data from metadata
    print(f"Reading training data from {Config.TRAIN_PATH}...")
    df_train = pd.read_parquet(Config.TRAIN_PATH)

    print(f"Reading test data from {Config.TEST_PATH}...")
    df_test = pd.read_parquet(Config.TEST_PATH)

    # Separate Targets and IDs
    y_train_raw = df_train[Config.TARGET_COL].values
    test_ids = df_test[Config.ID_COL].values

    # Drop non-feature columns
    # Train: Drop Id and Target
    X_train = df_train.drop(columns=[Config.ID_COL, Config.TARGET_COL])
    # Test: Drop Id
    X_test = df_test.drop(columns=[Config.ID_COL])

    # Apply Feature Engineering
    X_train = add_interaction_features(X_train)
    X_test = add_interaction_features(X_test)

    # Encode Targets (Map to 0..N-1)
    class_map = get_class_map()
    # Use numpy vectorize for efficient mapping
    mapper = np.vectorize(class_map.get)
    y_train = mapper(y_train_raw)

    # Save to Cache
    print(f"Saving processed data to cache: {Config.WORKING_DIR}")
    X_train.to_parquet(Config.CACHE_TRAIN_X, index=False)
    np.save(Config.CACHE_TRAIN_Y, y_train)
    X_test.to_parquet(Config.CACHE_TEST_X, index=False)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    return X_train, y_train, X_test, test_ids
