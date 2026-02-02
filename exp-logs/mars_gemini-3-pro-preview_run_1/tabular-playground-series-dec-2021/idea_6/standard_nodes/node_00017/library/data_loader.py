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
    Otherwise, loads from metadata CSVs, reduces memory, and caches.

    Updated to return Train, Val, and Test separately to support Hold-Out Ensemble validation.
    Cite solution_lesson_node_00016.

    Returns:
        df_train (pd.DataFrame): The training dataset.
        df_val (pd.DataFrame): The validation dataset.
        df_test (pd.DataFrame): The test dataset.
    """

    # 1. Check Cache
    if load_cached_data:
        if (
            os.path.exists(config.CACHE_TRAIN_PATH)
            and os.path.exists(config.CACHE_VAL_PATH)
            and os.path.exists(config.CACHE_TEST_PATH)
        ):
            print(f"Loading data from cache: {config.WORKING_DIR}")
            df_train = pd.read_parquet(config.CACHE_TRAIN_PATH)
            df_val = pd.read_parquet(config.CACHE_VAL_PATH)
            df_test = pd.read_parquet(config.CACHE_TEST_PATH)
            return df_train, df_val, df_test
        else:
            print("Cache not found or incomplete. Loading from raw metadata...")
    else:
        print("Ignoring cache. Loading from raw metadata...")

    # 2. Load from Metadata CSVs
    print(f"Reading {config.TRAIN_METADATA_PATH}...")
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)

    print(f"Reading {config.VAL_METADATA_PATH}...")
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    print(f"Reading {config.TEST_METADATA_PATH}...")
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # 3. Reduce Memory Usage
    print("Reducing memory usage...")
    df_train = reduce_memory_usage(df_train)
    df_val = reduce_memory_usage(df_val)
    df_test = reduce_memory_usage(df_test)

    # 4. Handle Debug Mode
    if config.DEBUG:
        print(f"DEBUG mode enabled. Sampling {config.DEBUG_SAMPLE_SIZE} rows...")
        df_train = df_train.sample(
            n=min(len(df_train), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), config.DEBUG_SAMPLE_SIZE // 5), random_state=config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        ).reset_index(drop=True)

    # 5. Save to Cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Saving processed data to cache: {config.WORKING_DIR}")
    df_train.to_parquet(config.CACHE_TRAIN_PATH, index=False)
    df_val.to_parquet(config.CACHE_VAL_PATH, index=False)
    df_test.to_parquet(config.CACHE_TEST_PATH, index=False)

    return df_train, df_val, df_test
