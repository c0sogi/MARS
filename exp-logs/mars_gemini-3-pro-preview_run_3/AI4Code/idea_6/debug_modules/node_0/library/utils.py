import os
import json
import random
import numpy as np
import pandas as pd
import torch
from bisect import bisect_left
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_notebook(file_path):
    """
    Reads a notebook JSON file and returns the parsed data.

    Args:
        file_path (str): Relative path to the notebook file (e.g., 'train/00001756c60be8.json').

    Returns:
        dict: A dictionary containing 'cell_type' and 'source' keys.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def count_inversions(a):
    """
    Counts the number of inversions in a list using bisect (O(N log N)).
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    This is equivalent to the number of swaps needed to sort the array.

    Args:
        a (list): A list of integers (ranks).

    Returns:
        int: The number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to keep sorted_so_far sorted
        idx = bisect_left(sorted_so_far, x)
        # All elements to the right of idx in sorted_so_far are greater than x
        # and appeared earlier in the sequence, so they form inversions with x.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau_metric(df_pred, df_gt):
    """
    Calculates the Kendall tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_pred (pd.DataFrame): DataFrame with 'id' and 'cell_order' (space-delimited string).
        df_gt (pd.DataFrame): DataFrame with 'id' and 'cell_order' (space-delimited string).

    Returns:
        float: The Kendall tau score.
    """
    # Merge predictions with ground truth on 'id' to ensure alignment
    df = pd.merge(df_pred, df_gt, on="id", suffixes=("_pred", "_gt"))

    total_swaps = 0
    total_possible = 0

    for _, row in df.iterrows():
        pred_order = row["cell_order_pred"].split()
        gt_order = row["cell_order_gt"].split()

        n = len(gt_order)
        # If there are fewer than 2 items, no sorting is needed; contribution to denominator is 0.
        if n < 2:
            continue

        # Map ground truth cell IDs to their correct rank (0, 1, 2, ...)
        gt_ranks = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert the predicted cell ID sequence to a rank sequence.
        # Filter to ensure we only consider cells present in the ground truth.
        pred_ranks = [
            gt_ranks[cell_id] for cell_id in pred_order if cell_id in gt_ranks
        ]

        # The number of swaps to sort pred_ranks into [0, 1, 2, ...] is the number of inversions.
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    if total_possible == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score
