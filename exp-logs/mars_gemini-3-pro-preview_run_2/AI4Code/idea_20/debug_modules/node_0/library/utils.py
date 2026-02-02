import os
import pandas as pd
from bisect import bisect_left
from typing import List, Optional
from library.config import Config


def count_inversions(a: List[int]) -> int:
    """
    Counts the number of inversions in a list using a bisect-based approach.
    An inversion is a pair of indices (i, j) such that i < j and a[i] > a[j].
    This corresponds to the number of swaps required to sort the array.

    Args:
        a: A list of integers (ranks).

    Returns:
        The total count of inversions.
    """
    inversions = 0
    sorted_so_far = []
    # Iterate backwards through the list
    for x in reversed(a):
        # Find the position where x should be inserted to maintain order
        # The index returned by bisect_left represents the number of elements
        # currently in sorted_so_far that are strictly less than x.
        # Since we are iterating backwards, these elements appeared to the RIGHT of x
        # in the original list. Thus, they form inversions with x.
        idx = bisect_left(sorted_so_far, x)
        inversions += idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau(ground_truths: List[List[str]], predictions: List[List[str]]) -> float:
    """
    Computes the Kendall Tau correlation metric accumulated across the collection.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n * (n - 1))

    Args:
        ground_truths: List of lists, where each inner list contains the correct ordered cell_ids.
        predictions: List of lists, where each inner list contains the predicted ordered cell_ids.

    Returns:
        The global Kendall Tau score.
    """
    total_swaps = 0
    total_pairs = 0

    if len(ground_truths) != len(predictions):
        raise ValueError(
            f"Length mismatch between ground truth ({len(ground_truths)}) and predictions ({len(predictions)})"
        )

    for gt, pred in zip(ground_truths, predictions):
        n = len(gt)

        # If a notebook has 0 or 1 cell, it contributes 0 pairs to the denominator.
        if n <= 1:
            continue

        # Map cell_ids to their ground truth rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(gt)}

        # Convert prediction sequence to rank sequence
        # We assume the prediction contains exactly the same set of IDs as the ground truth.
        try:
            pred_ranks = [rank_map[cell_id] for cell_id in pred]
        except KeyError as e:
            # In a strict pipeline, this indicates a malformed prediction (missing/extra IDs)
            raise ValueError(
                f"Predicted cell ID {e} not found in ground truth for a notebook."
            )

        # Calculate swaps (inversions) needed to sort the predicted ranks
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_pairs += n * (n - 1)

    # Avoid division by zero if dataset is empty or contains only trivial notebooks
    if total_pairs == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_pairs)
    return score


def format_submission(
    ids: List[str], cell_orders: List[List[str]], output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Formats predictions into the required submission DataFrame and optionally saves to CSV.

    Args:
        ids: List of notebook IDs.
        cell_orders: List of lists, where each inner list is the ordered sequence of cell_ids.
        output_path: Optional file path to save the submission CSV.

    Returns:
        A pandas DataFrame with columns ['id', 'cell_order'].
    """
    # Join cell IDs with a single space
    joined_orders = [" ".join(order) for order in cell_orders]

    df = pd.DataFrame({"id": ids, "cell_order": joined_orders})

    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
