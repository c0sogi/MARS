import os
import json
import random
import numpy as np
import pandas as pd
import torch
from bisect import bisect
from library import config


def set_seed(seed=config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_message(message):
    """
    Prints a message to stdout.
    """
    print(message)


def read_notebook_json(filepath):
    """
    Reads a notebook JSON file and returns its content.

    Args:
        filepath (str): Full path to the JSON file.

    Returns:
        dict: The parsed JSON content.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        log_message(f"Error reading file {filepath}: {e}")
        return {}


def count_inversions(a):
    """
    Counts the number of inversions in a list using a bisect approach.
    Complexity: O(N log N)
    """
    inversions = 0
    sorted_list = []
    for x in a:
        # Find the position where x should be inserted to keep the list sorted
        idx = bisect(sorted_list, x)
        # The number of elements to the right of this position are greater than x
        # and have appeared before x, thus forming inversions.
        inversions += len(sorted_list) - idx
        sorted_list.insert(idx, x)
    return inversions


def kendall_tau_metric(df_true, df_pred):
    """
    Calculates the Kendall Tau correlation metric as defined in the task.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_true (pd.DataFrame): DataFrame with columns ['id', 'cell_order'] (Ground Truth).
        df_pred (pd.DataFrame): DataFrame with columns ['id', 'cell_order'] (Predictions).

    Returns:
        float: The Kendall Tau score.
    """
    # Merge on ID to ensure alignment
    df = df_true.merge(df_pred, on="id", suffixes=("_true", "_pred"))

    total_swaps = 0
    total_possible_pairs = 0

    for _, row in df.iterrows():
        true_order = row["cell_order_true"].split()
        pred_order = row["cell_order_pred"].split()

        n = len(true_order)
        if n < 2:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(true_order)}

        # Convert the predicted order into a list of these ranks
        # If a predicted cell is not in the true order (should not happen in this task),
        # we can ignore it or handle error. Here we assume strict matching sets.
        pred_ranks = [
            rank_map[cell_id] for cell_id in pred_order if cell_id in rank_map
        ]

        # Calculate swaps (inversions) needed to sort pred_ranks
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible_pairs += n * (n - 1)

    if total_possible_pairs == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible_pairs)
    return score
