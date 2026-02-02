import os
import random
import bisect
import numpy as np
import torch
from typing import Dict, List, Set


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
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


def count_inversions(a: List[int]) -> int:
    """
    Counts the number of inversions in a list of integers using a bisect-based approach.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].

    Args:
        a (List[int]): The list of integers (ranks).

    Returns:
        int: The number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect_right returns the insertion point after any existing entries of x.
        # Elements currently in sorted_so_far to the right of this index are
        # greater than x and were seen previously, thus forming inversions.
        idx = bisect.bisect_right(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(
    ground_truth: Dict[str, List[str]], predictions: Dict[str, List[str]]
) -> float:
    """
    Computes the Kendall tau correlation metric accumulated across the collection.

    Formula: K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))
    Where S_i is the number of swaps needed to sort the prediction to the ground truth.

    Args:
        ground_truth: Dictionary mapping notebook_id to the correct list of cell_ids.
        predictions: Dictionary mapping notebook_id to the predicted list of cell_ids.

    Returns:
        float: The global Kendall tau score.
    """
    total_swaps = 0
    total_possible = 0

    # Intersect keys to ensure we only compare notebooks present in both sets
    common_ids = set(ground_truth.keys()) & set(predictions.keys())

    for nb_id in common_ids:
        gt_order = ground_truth[nb_id]
        pred_order = predictions[nb_id]

        n = len(gt_order)

        # If a notebook has 0 or 1 cell, no ordering is needed, and no pairs exist.
        if n <= 1:
            continue

        # Map ground truth cell IDs to their rank (0 to n-1)
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert the predicted cell order into a list of ranks based on ground truth.
        # We filter to ensure we only consider cells that exist in the ground truth
        # (handling potential mismatches gracefully, though they shouldn't occur in clean data).
        ranks = []
        for cell_id in pred_order:
            if cell_id in gt_rank_map:
                ranks.append(gt_rank_map[cell_id])

        # The number of swaps to sort the prediction is equal to the number of inversions
        # in the rank sequence.
        swaps = count_inversions(ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    # Avoid division by zero if all notebooks are trivial
    if total_possible == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score
