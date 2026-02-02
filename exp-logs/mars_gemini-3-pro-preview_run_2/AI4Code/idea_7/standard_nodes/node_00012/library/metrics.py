import pandas as pd
from bisect import bisect_right
from typing import List, Dict, Union
from library.config import Config


def count_inversions(prediction: List[str], ground_truth: List[str]) -> int:
    """
    Counts the number of swaps (inversions) needed to transform the prediction
    into the ground truth order.

    Args:
        prediction: List of cell IDs in the predicted order.
        ground_truth: List of cell IDs in the correct order.

    Returns:
        int: The number of inversions (swaps).
    """
    # Map ground truth IDs to their target rank (0 to n-1)
    rank_map = {cell_id: i for i, cell_id in enumerate(ground_truth)}

    # Convert prediction to a list of ranks
    # We filter to ensure we only consider IDs present in both lists to avoid key errors
    # In a valid submission, sets of IDs should be identical.
    pred_ranks = [rank_map[cell_id] for cell_id in prediction if cell_id in rank_map]

    # Count inversions using a Fenwick-tree-like approach with bisect (insertion sort logic)
    # We iterate through the predicted ranks and for each element, count how many
    # elements seen so far are greater than it.
    inversions = 0
    sorted_seen = []

    for rank in pred_ranks:
        # bisect_right returns the insertion point i such that all e in sorted_seen[:i] <= rank
        idx = bisect_right(sorted_seen, rank)

        # Elements in sorted_seen[idx:] are strictly greater than 'rank'
        # These elements appear before 'rank' in prediction but are larger, thus they form inversions.
        inversions += len(sorted_seen) - idx

        # Insert rank to maintain sorted property for next iterations
        sorted_seen.insert(idx, rank)

    return inversions


def score_dataframe(val_df: pd.DataFrame, predictions: Dict[str, List[str]]) -> float:
    """
    Computes the global Kendall Tau correlation for a validation dataframe according
    to the competition metric.

    Metric Formula: K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))

    Args:
        val_df: DataFrame containing at least 'id' and 'cell_order' columns.
                'cell_order' should be a space-delimited string of cell IDs.
        predictions: Dictionary mapping notebook 'id' to a list of cell IDs (predicted order).

    Returns:
        float: The global Kendall Tau score.
    """
    total_swaps = 0
    total_max_swaps_term = 0  # This represents Sum(n_i * (n_i - 1))

    for _, row in val_df.iterrows():
        nb_id = row["id"]
        gt_order_str = row["cell_order"]

        # Skip if no prediction available for this ID
        if nb_id not in predictions:
            continue

        # Parse ground truth
        if isinstance(gt_order_str, str):
            gt_order = gt_order_str.split()
        else:
            gt_order = list(gt_order_str)

        pred_order = predictions[nb_id]

        # Ensure prediction is a list
        if isinstance(pred_order, str):
            pred_order = pred_order.split()

        n = len(gt_order)

        # If a notebook has 0 or 1 cell, it has 0 possible swaps and contributes 0 to denominator.
        if n <= 1:
            continue

        swaps = count_inversions(pred_order, gt_order)

        total_swaps += swaps
        total_max_swaps_term += n * (n - 1)

    # Avoid division by zero if dataset is empty or contains only trivial notebooks
    if total_max_swaps_term == 0:
        return 1.0

    kendall_tau = 1 - 4 * (total_swaps / total_max_swaps_term)

    return kendall_tau
