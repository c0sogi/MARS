import os
import random
import time
import numpy as np
import torch
import pandas as pd
from contextlib import contextmanager


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    return device


def load_metadata(split):
    """
    Loads the metadata CSV for a specific split (train, val, test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    path = os.path.join("./metadata", f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def get_feature_intersection(df_train, df_test, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test dataframes
    to prevent feature leakage.

    Args:
        df_train (pd.DataFrame): Training dataframe.
        df_test (pd.DataFrame): Test dataframe.
        exclude_cols (list, optional): List of columns to explicitly exclude
                                     (e.g., target variables).

    Returns:
        list: Sorted list of common column names.
    """
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    intersection = train_cols.intersection(test_cols)

    if exclude_cols:
        intersection = intersection - set(exclude_cols)

    return sorted(list(intersection))


def save_submission(
    request_ids, predictions, output_path="./submission/submission.csv"
):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        request_ids (iterable): List or array of request_ids.
        predictions (iterable): List or array of predicted probabilities.
        output_path (str): Path to save the submission file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


@contextmanager
def Timer(name):
    """
    Context manager to measure and print execution time of a block.
    """
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[{name}] done in {t1 - t0:.2f} s")


def load_or_compute(
    filename,
    compute_func,
    load_cached_data,
    cache_dir="./working/idea_25/",
    file_type="parquet",
    expected_len=None,
    **kwargs,
):
    """
    Generic caching utility. Loads data from disk if it exists and loading is enabled.
    Otherwise, runs the compute_func, saves the result, and returns it.

    Args:
        filename (str): Name of the file (e.g., 'features.parquet').
        compute_func (callable): Function to compute the data if cache is missed.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_dir (str): Directory to store cached files.
        file_type (str): 'parquet', 'npy', or 'npz'.
        expected_len (int, optional): Expected number of rows/samples. If cached data
                                      mismatch, it triggers recomputation.
        **kwargs: Arguments passed to compute_func.

    Returns:
        The loaded or computed data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)

    # Try loading
    if load_cached_data and os.path.exists(filepath):
        print(f"Loading cached data from {filepath}...")
        try:
            data = None
            if file_type == "parquet":
                data = pd.read_parquet(filepath)
            elif file_type == "npy":
                data = np.load(filepath)
            elif file_type == "npz":
                data = np.load(filepath)
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")

            # Validation: Check dimensions if expected_len is provided
            if expected_len is not None and data is not None:
                # For npz, len() usually returns number of arrays/files, not samples, so skip
                if file_type != "npz":
                    if len(data) != expected_len:
                        print(
                            f"Cache mismatch for {filename}: Expected {expected_len} rows, found {len(data)}. Recomputing..."
                        )
                        data = None

            if data is not None:
                return data

        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute
    print(f"Computing data for {filename}...")
    data = compute_func(**kwargs)

    # Save
    print(f"Saving data to {filepath}...")
    if file_type == "parquet":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Computed data must be a DataFrame for parquet format.")
        data.to_parquet(filepath, index=False)
    elif file_type == "npy":
        np.save(filepath, data)
    elif file_type == "npz":
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            # If data is a tuple/list of arrays, save as args
            if isinstance(data, (list, tuple)):
                np.savez(filepath, *data)
            else:
                raise TypeError("For npz, data must be a dict or list/tuple of arrays.")
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    return data
