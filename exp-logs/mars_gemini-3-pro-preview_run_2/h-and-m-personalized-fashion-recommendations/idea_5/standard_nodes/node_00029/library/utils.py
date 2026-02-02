import os
import random
import numpy as np
import torch
import pandas as pd
import functools
import gc
import warnings
from scipy import sparse
from pathlib import Path
from library import config

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def calculate_map12(val_df, sub_df):
    """
    Calculate MAP@12 for the validation set.

    Args:
        val_df: DataFrame containing 'customer_id' and 'article_id' (ground truth).
        sub_df: DataFrame containing 'customer_id' and 'prediction' (space-separated string).

    Returns:
        float: The MAP@12 score.
    """
    # Ensure IDs are strings
    val_df = val_df.copy()
    sub_df = sub_df.copy()
    val_df["customer_id"] = val_df["customer_id"].astype(str)
    val_df["article_id"] = val_df["article_id"].astype(str)
    sub_df["customer_id"] = sub_df["customer_id"].astype(str)
    sub_df["prediction"] = sub_df["prediction"].astype(str)

    # Group ground truth by customer
    # We assume val_df contains transactions.
    # We aggregate article_ids into a list for each customer.
    truth = val_df.groupby("customer_id")["article_id"].apply(list).reset_index()
    truth.columns = ["customer_id", "actual"]

    # Merge predictions with ground truth
    # We only score customers who are in the validation set (truth)
    merged = truth.merge(sub_df, on="customer_id", how="left")

    # Fill missing predictions
    merged["prediction"] = merged["prediction"].fillna("")

    aps = []

    # Iterate through each customer
    for _, row in merged.iterrows():
        actual = row["actual"]
        predicted_str = row["prediction"]

        if not actual:
            continue

        # Parse predictions (top 12)
        predicted = predicted_str.split()[:12]

        score = 0.0
        num_hits = 0.0

        # Create a set for O(1) lookup of relevant items
        actual_set = set(actual)

        for i, p in enumerate(predicted):
            if p in actual_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # m is the number of ground truth values per customer
        m = len(actual)
        ap = score / min(m, 12)
        aps.append(ap)

    map12 = np.mean(aps) if aps else 0.0
    print(f"MAP@12: {map12}")
    return map12


def cache_result(file_path):
    """
    Decorator to cache function results to a file based on 'load_cached_data' argument.

    Args:
        file_path (str or Path): The path to the cache file.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            load_cache = kwargs.get("load_cached_data", False)
            path = Path(file_path)

            # Ensure output directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Try to load if requested and exists
            if load_cache and path.exists():
                print(f"Loading cached result from {path}...")
                ext = path.suffix
                try:
                    if ext == ".parquet":
                        return pd.read_parquet(path)
                    elif ext == ".npy":
                        return np.load(path, allow_pickle=True)
                    elif ext == ".npz":
                        # Assuming sparse matrix for .npz in this pipeline
                        return sparse.load_npz(path)
                    elif ext == ".pt":
                        return torch.load(path, map_location="cpu", weights_only=False)
                    else:
                        print(
                            f"Warning: Unsupported extension {ext} for loading, recomputing..."
                        )
                except Exception as e:
                    print(f"Error loading cache {path}: {e}. Recomputing...")

            # Compute
            result = func(*args, **kwargs)

            # Save
            print(f"Saving result to {path}...")
            ext = path.suffix
            try:
                if ext == ".parquet":
                    if isinstance(result, pd.DataFrame):
                        result.to_parquet(path, index=False)
                    else:
                        raise ValueError(
                            "Result is not a DataFrame, cannot save as parquet."
                        )
                elif ext == ".npy":
                    np.save(path, result)
                elif ext == ".npz":
                    if sparse.issparse(result):
                        sparse.save_npz(path, result)
                    else:
                        # Fallback for dict of arrays
                        np.savez(path, **result)
                elif ext == ".pt":
                    torch.save(result, path)
                else:
                    print(f"Warning: Unsupported extension {ext} for saving.")
            except Exception as e:
                print(f"Error saving cache {path}: {e}")

            return result

        return wrapper

    return decorator
