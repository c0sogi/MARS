import os
import pandas as pd
import numpy as np
from library import config


def reduce_memory_usage(df, verbose=True):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # Use float32 for stability, float16 can be risky with accumulations
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f"Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def load_dataset(load_cached_data=True):
    """
    Loads the dataset.
    If load_cached_data is True and cache exists, loads from parquet.
    Otherwise, loads from metadata CSVs, concatenates train/val, reduces memory, and caches.

    Returns:
        df_train (pd.DataFrame): The full training dataset (Train + Val).
        df_test (pd.DataFrame): The test dataset.
    """

    # 1. Check Cache
    if load_cached_data:
        if os.path.exists(config.CACHE_TRAIN_PATH) and os.path.exists(
            config.CACHE_TEST_PATH
        ):
            print(f"Loading data from cache: {config.WORKING_DIR}")
            df_train = pd.read_parquet(config.CACHE_TRAIN_PATH)
            df_test = pd.read_parquet(config.CACHE_TEST_PATH)
            return df_train, df_test
        else:
            print("Cache not found or incomplete. Loading from raw metadata...")
    else:
        print("Ignoring cache. Loading from raw metadata...")

    # 2. Load from Metadata CSVs
    # We combine train and val metadata to form the full training set for Cross-Validation
    print(f"Reading {config.TRAIN_METADATA_PATH}...")
    df_train_part = pd.read_csv(config.TRAIN_METADATA_PATH)

    print(f"Reading {config.VAL_METADATA_PATH}...")
    df_val_part = pd.read_csv(config.VAL_METADATA_PATH)

    print(f"Reading {config.TEST_METADATA_PATH}...")
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Concatenate Train and Val
    print("Concatenating Train and Validation sets for full training data...")
    df_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Clean up temporary dataframes to free memory
    del df_train_part, df_val_part

    # 3. Reduce Memory Usage
    print("Reducing memory usage for Training set...")
    df_train = reduce_memory_usage(df_train)
    print("Reducing memory usage for Test set...")
    df_test = reduce_memory_usage(df_test)

    # 4. Handle Debug Mode
    if config.DEBUG:
        print(f"DEBUG mode enabled. Sampling {config.DEBUG_SAMPLE_SIZE} rows...")
        # Random sample for debug speed
        df_train = df_train.sample(
            n=min(len(df_train), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        ).reset_index(drop=True)

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Saving processed data to cache: {config.WORKING_DIR}")
    # Use Parquet for efficient storage
    df_train.to_parquet(config.CACHE_TRAIN_PATH, index=False)
    df_test.to_parquet(config.CACHE_TEST_PATH, index=False)

    return df_train, df_test
