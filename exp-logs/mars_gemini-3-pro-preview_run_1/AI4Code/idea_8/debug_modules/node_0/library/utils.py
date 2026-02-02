import os
import random
import numpy as np
import torch
import pandas as pd
from bisect import bisect


def set_seed(seed=42):
    """
    Sets the seed for random, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers using bisect.
    Time Complexity: O(N log N)
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to keep the list sorted
        idx = bisect(sorted_so_far, x)
        # All elements currently in sorted_so_far at indices >= idx are greater than x
        # and appeared before x in the original list 'a', thus forming inversions.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(df_pred, df_gt):
    """
    Computes the global Kendall tau correlation metric defined in the task.

    K = 1 - 4 * (Sum of Inversions) / (Sum of n*(n-1))

    Args:
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns.
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns.

    Returns:
        float: The Kendall tau correlation score.
    """
    # Create dictionaries for fast lookup
    preds = dict(zip(df_pred["id"], df_pred["cell_order"]))
    gts = dict(zip(df_gt["id"], df_gt["cell_order"]))

    total_inversions = 0
    total_n_combination = 0

    # Iterate over the intersection of IDs (usually validation set)
    common_ids = set(preds.keys()).intersection(set(gts.keys()))

    for nb_id in common_ids:
        pred_order = preds[nb_id].split()
        gt_order = gts[nb_id].split()

        n = len(gt_order)
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Map the predicted order to the ground truth ranks
        # We filter to ensure we only consider cells present in the ground truth
        # (though in this task, pred and gt sets should be identical)
        pred_ranks = [
            gt_rank_map[cell_id] for cell_id in pred_order if cell_id in gt_rank_map
        ]

        # Count inversions needed to sort pred_ranks (which is equivalent to sorting pred_order to gt_order)
        s = count_inversions(pred_ranks)

        total_inversions += s
        total_n_combination += n * (n - 1)

    if total_n_combination == 0:
        return 1.0

    k = 1 - 4 * (total_inversions / total_n_combination)
    return k
