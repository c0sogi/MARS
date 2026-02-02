import os
import random
import numpy as np
import torch
import pandas as pd
from bisect import bisect


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_inversions(a):
    """
    Counts the number of inversions in a list. An inversion is a pair of
    elements (a[i], a[j]) such that i < j and a[i] > a[j].
    This is equivalent to the number of swaps required to sort the list.

    Args:
        a (list): A list of comparable elements (e.g., integers).

    Returns:
        int: The number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to keep the list sorted.
        # Elements to the right of this position in 'sorted_so_far' are
        # elements seen previously that are greater than x.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau_metric(df_pred, df_gt):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    Formula: K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))
    Where S_i is the number of swaps (inversions) needed to sort the predicted
    order into the ground truth order for notebook i, and n_i is the number of cells.

    Args:
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns.
                                'cell_order' should be a space-delimited string of cell IDs.
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns.
                              'cell_order' should be a space-delimited string of cell IDs.

    Returns:
        float: The calculated Kendall Tau score.
    """
    # Create copies to avoid modifying original dataframes
    preds = df_pred.copy()
    gt = df_gt.copy()

    # Merge on notebook ID to ensure we compare the correct notebooks
    # Using inner join to evaluate on the intersection of IDs
    df = pd.merge(preds, gt, on="id", suffixes=("_pred", "_gt"))

    total_inversions = 0
    total_pairs = 0

    for _, row in df.iterrows():
        pred_str = row["cell_order_pred"]
        gt_str = row["cell_order_gt"]

        # Parse space-delimited strings into lists
        pred_order = pred_str.split()
        gt_order = gt_str.split()

        n = len(gt_order)

        # If a notebook has 0 or 1 cell, it contributes nothing to the denominator
        # and has 0 inversions.
        if n < 2:
            continue

        # Map ground truth cell IDs to their correct rank (0, 1, 2, ...)
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert the predicted cell ID sequence into a sequence of ranks.
        # We only consider cells that exist in the ground truth.
        ranks = []
        for cell_id in pred_order:
            if cell_id in gt_rank_map:
                ranks.append(gt_rank_map[cell_id])

        # Calculate the number of swaps (inversions) needed to sort the predicted ranks
        s = count_inversions(ranks)

        total_inversions += s
        total_pairs += n * (n - 1)

    # Handle edge case where no valid pairs exist across all notebooks
    if total_pairs == 0:
        return 1.0

    score = 1 - 4 * (total_inversions / total_pairs)
    return score
