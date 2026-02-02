import os
import pandas as pd
import numpy as np
from library.config import Config


def load_essay_data(
    data_type: str,
    load_cached_data: bool = True,
    debug: bool = Config.DEBUG,
    debug_size: int = Config.DEBUG_SAMPLE_SIZE,
):
    """
    Loads essay data from metadata CSVs or cached Parquet files.

    Args:
        data_type (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, loads only a subset of data.
        debug_size (int): Number of rows to load in debug mode.

    Returns:
        tuple: (ids, texts, scores)
            - ids (list): List of essay IDs.
            - texts (list): List of full essay texts.
            - scores (np.ndarray or None): Array of scores (float) for train/val, None for test.
    """
    # 1. Determine Input Path
    if data_type == "train":
        source_path = Config.TRAIN_DATA_PATH
    elif data_type == "val":
        source_path = Config.VAL_DATA_PATH
    elif data_type == "test":
        source_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(
            f"Invalid data_type: {data_type}. Must be 'train', 'val', or 'test'."
        )

    # 2. Determine Cache Path
    # We append '_debug' to the filename if debug mode is active to separate caches
    suffix = "_debug" if debug else ""
    cache_filename = f"{data_type}_data{suffix}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    df = None

    # 3. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_path):
            try:
                print(f"Loading {data_type} data from cache: {cache_path}")
                df = pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache ({e}). Reloading from source.")
                df = None
        else:
            print(f"Cache not found for {data_type}. Loading from source.")

    # 4. Load from Source if needed
    if df is None:
        print(f"Loading {data_type} data from source: {source_path}")
        df = pd.read_csv(source_path)

        # Apply Debug Sampling
        if debug:
            print(f"Debug mode active. Sampling first {debug_size} rows.")
            df = df.head(debug_size)

        # Save to Cache
        print(f"Saving processed {data_type} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

    # 5. Extract and Format Data
    ids = df["essay_id"].tolist()
    texts = df["full_text"].tolist()

    if "score" in df.columns:
        # Convert to float for regression tasks
        scores = df["score"].values.astype(np.float32)
    else:
        scores = None

    print(f"Loaded {len(ids)} rows for {data_type}.")
    return ids, texts, scores
