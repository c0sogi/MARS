import os
import pandas as pd
from library.config import Config


def load_data(debug=Config.DEBUG, load_cached_data=True):
    """
    Loads train, validation, and test datasets from metadata or cache.

    Args:
        debug (bool): If True, returns a small subset of the data for debugging.
        load_cached_data (bool): If True, attempts to load from Parquet cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    datasets = {}
    # Map dataset keys to their source CSV path and cache filename
    files = {
        "train": (Config.TRAIN_PATH, "train.parquet"),
        "val": (Config.VAL_PATH, "val.parquet"),
        "test": (Config.TEST_PATH, "test.parquet"),
    }

    for key, (csv_path, parquet_name) in files.items():
        parquet_path = os.path.join(Config.CACHE_DIR, parquet_name)

        df = None
        # 1. IF load_cached_data is True: Try to load the file.
        if load_cached_data and os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                # If loading fails (corrupt file), we will reload from CSV
                df = None

        # 2. IF loading fails OR load_cached_data is False:
        if df is None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{key} file not found at {csv_path}")

            # Compute/process the data from scratch (Read CSV)
            df = pd.read_csv(csv_path)

            # Save the result to the cache directory
            df.to_parquet(parquet_path, index=False)

        datasets[key] = df

    train_df = datasets["train"]
    val_df = datasets["val"]
    test_df = datasets["test"]

    # Handle Debugging: Slice data if debug mode is active
    if debug:
        sample_size = Config.DEBUG_SAMPLE_SIZE
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
        test_df = test_df.head(sample_size)

    return train_df, val_df, test_df


def save_submission(submission_df):
    """
    Saves the prediction dataframe to a CSV file in the submission directory.

    Args:
        submission_df (pd.DataFrame): Dataframe containing 'id' and class probabilities.
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
