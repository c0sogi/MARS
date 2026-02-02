import os
import random
import numpy as np
import torch
import pandas as pd
from bisect import bisect_left
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers.
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
        # bisect_left returns the index where x should be inserted to maintain order.
        # Elements to the right of this index in sorted_so_far are greater than x.
        # Since those elements appeared before x in 'a', they form inversions with x.
        idx = bisect_left(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(df_pred: pd.DataFrame, df_gt: pd.DataFrame) -> float:
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))

    Where:
        S_i is the number of swaps (inversions) needed to sort the predicted order
            into the ground truth order for notebook i.
        n_i is the number of cells in notebook i.

    Args:
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).

    Returns:
        float: The computed Kendall Tau score.
    """
    # Ensure IDs are strings and merge
    df_pred = df_pred.copy()
    df_gt = df_gt.copy()

    df_pred["id"] = df_pred["id"].astype(str)
    df_gt["id"] = df_gt["id"].astype(str)

    # Merge on ID to ensure alignment
    df_merged = pd.merge(df_gt, df_pred, on="id", suffixes=("_gt", "_pred"))

    total_swaps = 0
    total_possible_pairs = 0

    for _, row in df_merged.iterrows():
        gt_order = row["cell_order_gt"].split()
        pred_order = row["cell_order_pred"].split()

        n = len(gt_order)

        # If there's 0 or 1 cell, no ordering is needed, and no pairs exist.
        if n <= 1:
            continue

        # Map cell IDs to their ground truth rank (0 to n-1)
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert prediction to a list of ranks
        # We filter to ensure we only consider cells present in the ground truth
        # (though typically pred and gt should contain the same set of cells)
        ranks = [
            gt_rank_map[cell_id] for cell_id in pred_order if cell_id in gt_rank_map
        ]

        # Calculate swaps (inversions)
        swaps = count_inversions(ranks)

        total_swaps += swaps
        total_possible_pairs += n * (n - 1)

    if total_possible_pairs == 0:
        return 1.0  # Perfect score if no pairs exist to be ordered

    score = 1 - 4 * (total_swaps / total_possible_pairs)
    return score
