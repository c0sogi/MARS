import os
import json
import random
import numpy as np
import pandas as pd
import torch
from bisect import bisect, insort
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_notebook_json(filepath):
    """
    Reads and parses a notebook JSON file from the input directory.

    Args:
        filepath (str): The relative path to the JSON file (e.g., 'train/00001756c60be8.json').

    Returns:
        dict: The parsed JSON content of the notebook.
    """
    full_path = os.path.join(Config.INPUT_DIR, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # In a real pipeline, we might want to log this or raise, but for now we print
        print(f"Error reading {full_path}: {e}")
        return {}


def count_inversions(a):
    """
    Counts the number of inversions in a list using a bisect-based approach.
    An inversion is a pair of indices (i, j) such that i < j and a[i] > a[j].

    Args:
        a (list): A list of comparable elements (e.g., integers representing ranks).

    Returns:
        int: The total number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect finds the insertion point to maintain sorted order.
        # Elements currently in sorted_so_far that are to the right of this index
        # are strictly greater than x. Since they were processed before x,
        # they form inversions with x.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        insort(sorted_so_far, x)
    return inversions


def kendall_tau_metric(df_pred, df_gt):
    """
    Calculates the Kendall Tau correlation metric as defined for the AI4Code competition.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).

    Returns:
        float: The Kendall Tau score accumulated across the collection.
    """
    # Merge predictions with ground truth on 'id' to ensure alignment
    merged_df = pd.merge(df_pred, df_gt, on="id", suffixes=("_pred", "_gt"))

    total_swaps = 0
    total_possible_swaps = 0

    for _, row in merged_df.iterrows():
        pred_order = row["cell_order_pred"].split()
        gt_order = row["cell_order_gt"].split()

        n = len(gt_order)
        # If there is 0 or 1 cell, no ordering is needed, contribution to denominator is 0.
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert the predicted cell IDs into their ground truth ranks.
        # We filter to ensure we only consider cells present in the ground truth
        # (though valid predictions should match exactly).
        pred_ranks = [
            gt_rank_map[cell_id] for cell_id in pred_order if cell_id in gt_rank_map
        ]

        # The number of swaps to sort the prediction into the ground truth
        # is equivalent to the number of inversions in the list of ranks.
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible_swaps += n * (n - 1)

    if total_possible_swaps == 0:
        return 1.0  # Default to perfect score if no swaps were possible/needed

    score = 1 - 4 * (total_swaps / total_possible_swaps)
    return score


def load_or_save_cache(file_name, data_producer_fn, load_cached_data=True, **kwargs):
    """
    Generic caching utility to load data from disk or generate and save it if missing.
    Strictly follows the requirement: Try load -> If fail/disabled -> Generate -> Save -> Return.
    Uses Parquet for DataFrames and NPY for NumPy arrays.

    Args:
        file_name (str): The filename (e.g., 'data.parquet') to store in Config.WORKING_DIR.
        data_producer_fn (callable): Function to call to generate data if cache is missed.
        load_cached_data (bool): If True, attempts to load from disk first.
        **kwargs: Arguments passed to data_producer_fn.

    Returns:
        The loaded or generated data object.
    """
    cache_path = os.path.join(Config.WORKING_DIR, file_name)

    # 1. Try to load
    if load_cached_data and os.path.exists(cache_path):
        try:
            if file_name.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif file_name.endswith(".npy"):
                return np.load(cache_path, allow_pickle=True)
            else:
                # Fallback or error for unsupported types in this specific utility
                print(
                    f"Warning: Unsupported cache file extension for {file_name}. Regenerating."
                )
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Regenerating...")

    # 2. Generate data
    data = data_producer_fn(**kwargs)

    # 3. Save data
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    try:
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path)
        elif isinstance(data, np.ndarray):
            np.save(cache_path, data)
        else:
            print(
                f"Warning: Data type {type(data)} not automatically cacheable by this utility (only DataFrame/ndarray). Data not saved."
            )
    except Exception as e:
        print(f"Error saving cache to {cache_path}: {e}")

    return data
