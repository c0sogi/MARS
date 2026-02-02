import numpy as np
from bisect import bisect_right
from typing import List, Union, Any


def count_inversions(a: List[int]) -> int:
    """
    Counts the number of inversions in a list of integers using a bisect approach.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].

    This effectively calculates the minimum number of swaps needed to sort the array
    into ascending order (Bubble Sort distance).

    Args:
        a: List of integers representing ranks.

    Returns:
        The total number of inversions.
    """
    inversions = 0
    sorted_list = []
    for x in a:
        # Find the insertion point to maintain sorted order
        i = bisect_right(sorted_list, x)
        # The number of elements strictly greater than x that have already been
        # processed is equal to the number of elements to the right of the insertion point.
        # These elements appear before x in the input but are larger, forming inversions.
        inversions += len(sorted_list) - i
        sorted_list.insert(i, x)
    return inversions


def kendall_tau(ground_truths: List[List[str]], predictions: List[List[str]]) -> float:
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Where 'swaps' is the number of adjacent swaps needed to sort the predicted
    order into the ground truth order.

    Args:
        ground_truths: List of lists, where each inner list contains the correct ordered cell_ids.
        predictions: List of lists, where each inner list contains the predicted ordered cell_ids.

    Returns:
        The global Kendall Tau score accumulated across all notebooks.
    """
    total_swaps = 0
    total_possible = 0

    for gt, pred in zip(ground_truths, predictions):
        n = len(gt)
        # If there's 0 or 1 cell, no ordering is needed, and n*(n-1) is 0.
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1) based on ground truth
        rank_map = {cell_id: i for i, cell_id in enumerate(gt)}

        # Convert the predicted cell ID sequence into a sequence of ground truth ranks.
        # We filter to ensure we only process IDs present in the ground truth
        # (handling potential mismatches gracefully, though datasets should be consistent).
        pred_ranks = [rank_map[cell_id] for cell_id in pred if cell_id in rank_map]

        # The number of swaps to sort 'pred_ranks' is exactly the number of inversions.
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    # Avoid division by zero if all notebooks are trivial
    if total_possible == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score


def validate_ranks(ranks: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Ensures predicted normalized ranks fall within the valid bounds [0, 1].
    Values outside this range are clipped.

    Args:
        ranks: Array or list of predicted rank values (floats).

    Returns:
        Numpy array of validated ranks clipped to [0, 1].
    """
    ranks_arr = np.array(ranks)
    return np.clip(ranks_arr, 0.0, 1.0)
