import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dataset(split="train", sample_size=None):
    """
    Loads the dataset for the specified split from the metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test'.
        sample_size (int, optional): If provided, returns a random sample of this size.
                                     Useful for debugging/testing pipelines.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if split == "train":
        path = Config.TRAIN_CSV
    elif split == "val":
        path = Config.VAL_CSV
    elif split == "test":
        path = Config.TEST_CSV
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{split} dataset not found at {path}")

    # Load data
    df = pd.read_csv(path)

    # Debugging/Testing: Sample the data if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=Config.RANDOM_SEED).reset_index(
            drop=True
        )

    return df


def save_submission(predictions, filename=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (pd.DataFrame or dict):
            If DataFrame, must have columns [request_id, requester_received_pizza].
            If dict, must be {request_id: probability}.
        filename (str): Path to save the submission file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if isinstance(predictions, dict):
        # Convert dict to DataFrame
        df = pd.DataFrame(
            list(predictions.items()), columns=[Config.ID_COL, Config.TARGET_COL]
        )
    elif isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
        # Ensure correct columns
        if Config.ID_COL not in df.columns or Config.TARGET_COL not in df.columns:
            raise ValueError(
                f"DataFrame must contain columns: {Config.ID_COL}, {Config.TARGET_COL}"
            )
        df = df[[Config.ID_COL, Config.TARGET_COL]]
    else:
        raise TypeError("predictions must be a pandas DataFrame or a dictionary.")

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def get_cached_data(func, cache_path, load_cached_data=True, **kwargs):
    """
    Generic caching mechanism for deterministic data processing.

    Logic:
    1. IF load_cached_data is True: Try to load the file.
    2. IF loading fails OR load_cached_data is False:
       - Compute/process the data using `func`.
       - Save the result to `cache_path`.
    3. Return the data.

    Args:
        func (callable): The function to execute if cache is not found/used.
                         Must return an object that can be saved (DataFrame or numpy array).
        cache_path (str): Path to the cache file (must end in .parquet or .npy).
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments to pass to `func`.

    Returns:
        The data, either loaded from cache or computed by func.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load
    if load_cached_data and os.path.exists(cache_path):
        try:
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path)
            elif cache_path.endswith(".npz"):
                return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute
    data = func(**kwargs)

    # 3. Save
    if isinstance(data, pd.DataFrame):
        data.to_parquet(cache_path, index=False)
    elif isinstance(data, np.ndarray):
        np.save(cache_path, data)
    elif isinstance(data, dict) and cache_path.endswith(".npz"):
        # Support for saving dictionary of arrays to npz
        np.savez(cache_path, **data)
    else:
        # If we can't automatically determine how to save, we proceed but warn or fail
        # depending on strictness. Here we assume the user aligns return type with extension.
        pass

    return data
