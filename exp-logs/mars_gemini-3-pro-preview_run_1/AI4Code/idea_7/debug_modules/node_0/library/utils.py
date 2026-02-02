import os
import json
import random
import bisect
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_notebook(filepath):
    """
    Reads a notebook JSON file from the input directory.

    Args:
        filepath (str): Relative path to the notebook file (e.g., 'train/id.json').

    Returns:
        dict: The notebook content.
    """
    full_path = os.path.join(Config.INPUT_DIR, filepath)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_inversions(a):
    """
    Counts the number of inversions in a list using the bisect module.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    This represents the number of swaps needed to sort the array.

    Args:
        a (list): List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect_right returns the index where x should be inserted to maintain order.
        # Elements in sorted_so_far to the right of this index are greater than x.
        # Since they appeared before x, they form inversions with x.
        idx = bisect.bisect_right(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(df_gt, df_pred):
    """
    Computes the Kendall Tau correlation metric as defined in the task.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_gt (pd.DataFrame): DataFrame containing ground truth with columns ['id', 'cell_order'].
        df_pred (pd.DataFrame): DataFrame containing predictions with columns ['id', 'cell_order'].

    Returns:
        float: The Kendall Tau score.
    """
    # Merge on ID to ensure alignment
    df = df_gt.merge(df_pred, on="id", suffixes=("_gt", "_pred"))

    total_swaps = 0
    total_possible_pairs = 0

    for _, row in df.iterrows():
        gt_order = row["cell_order_gt"].split()
        pred_order = row["cell_order_pred"].split()

        n = len(gt_order)
        if n <= 1:
            continue

        # Map cell IDs to their correct rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert prediction to list of ranks
        # Filter to ensure we only consider cells present in both (though they should match)
        pred_ranks = [
            rank_map[cell_id] for cell_id in pred_order if cell_id in rank_map
        ]

        # Calculate swaps (inversions) for this notebook
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible_pairs += n * (n - 1)

    if total_possible_pairs == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_possible_pairs)
    return score
