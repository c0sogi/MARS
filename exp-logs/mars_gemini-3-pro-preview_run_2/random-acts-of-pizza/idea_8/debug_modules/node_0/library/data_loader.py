import os
import pandas as pd
from library.config import Config, get_raw_data


def load_datasets(load_cached_data=True, subsample_frac=None):
    """
    Loads the train, validation, and test datasets.

    This function handles the retrieval of raw data, merging with metadata,
    and caching the resulting DataFrames to Parquet files to improve performance
    on subsequent runs. It also supports subsampling for debugging purposes.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed DataFrames
                                 from the working directory cache.
        subsample_frac (float, optional): The fraction of data to return (0.0 < frac <= 1.0).
                                          Useful for quick debugging. Defaults to None.

    Returns:
        tuple: A tuple containing three pandas DataFrames: (df_train, df_val, df_test).
    """
    splits = ["train", "val", "test"]
    datasets = {}

    # Ensure the working directory for caching exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for split in splits:
        cache_path = os.path.join(Config.WORKING_DIR, f"{split}_raw_dataframe.parquet")
        df = None

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
            except Exception as e:
                print(
                    f"Failed to load cached {split} data: {e}. Reloading from source."
                )
                df = None

        # 2. If not in cache or cache load failed, load from source
        if df is None:
            df = get_raw_data(split=split)

            # Save to cache for future use
            try:
                df.to_parquet(cache_path, index=False)
            except Exception as e:
                print(f"Warning: Could not save {split} data to cache: {e}")

        # 3. Apply subsampling if requested
        if subsample_frac is not None and 0.0 < subsample_frac < 1.0:
            df = df.sample(
                frac=subsample_frac, random_state=Config.RANDOM_SEED
            ).reset_index(drop=True)

        datasets[split] = df

    return datasets["train"], datasets["val"], datasets["test"]
