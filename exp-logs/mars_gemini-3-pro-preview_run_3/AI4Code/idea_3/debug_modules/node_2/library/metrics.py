import pandas as pd
from bisect import bisect_right
from library.utils import set_seed


def count_inversions(prediction, ground_truth):
    """
    Calculates the number of swaps (inversions) needed to sort the predicted order
    into the ground truth order.

    Args:
        prediction (list or str): Predicted cell order (list of IDs or space-separated string).
        ground_truth (list or str): Ground truth cell order.

    Returns:
        int: Number of inversions (swaps).
    """
    # Normalize inputs to lists
    if isinstance(prediction, str):
        prediction = prediction.split()
    if isinstance(ground_truth, str):
        ground_truth = ground_truth.split()

    # Create a mapping from ground truth cell ID to its rank (0-indexed position)
    rank_map = {cell_id: i for i, cell_id in enumerate(ground_truth)}

    # Convert the predicted cell IDs to their corresponding ranks in the ground truth.
    # We filter out any IDs in prediction that aren't in ground_truth to be robust,
    # though valid predictions should be permutations.
    pred_ranks = [rank_map[cell_id] for cell_id in prediction if cell_id in rank_map]

    # Count inversions
    # An inversion is a pair of indices (i, j) such that i < j and rank[i] > rank[j].
    inversions = 0
    sorted_seen = []

    for rank in pred_ranks:
        # bisect_right returns the insertion point i such that all elements to the left are <= rank
        # and all elements to the right are > rank.
        # Since sorted_seen contains elements processed so far (to the left of current),
        # elements at indices >= idx are strictly greater than the current rank.
        idx = bisect_right(sorted_seen, rank)

        # The number of elements seen so far that are greater than the current rank
        num_greater = len(sorted_seen) - idx
        inversions += num_greater

        # Insert current rank into the sorted list to maintain order for future checks
        sorted_seen.insert(idx, rank)

    return inversions


def compute_kendall_tau(predictions, ground_truths):
    """
    Computes the global Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        predictions (dict or pd.DataFrame):
            - Dict: {id: cell_order_list_or_string}
            - DataFrame: Must contain columns ['id', 'cell_order']
        ground_truths (dict or pd.DataFrame):
            - Dict: {id: cell_order_list_or_string}
            - DataFrame: Must contain columns ['id', 'cell_order']

    Returns:
        float: The Kendall Tau score.
    """
    # Convert DataFrames to dictionaries for uniform processing
    if isinstance(predictions, pd.DataFrame):
        predictions = dict(zip(predictions["id"], predictions["cell_order"]))
    if isinstance(ground_truths, pd.DataFrame):
        ground_truths = dict(zip(ground_truths["id"], ground_truths["cell_order"]))

    total_swaps = 0
    total_possible_pairs = 0

    # Identify common notebooks to evaluate
    common_ids = set(predictions.keys()).intersection(set(ground_truths.keys()))

    for notebook_id in common_ids:
        pred_order = predictions[notebook_id]
        gt_order = ground_truths[notebook_id]

        # Determine n (number of cells) from the ground truth
        if isinstance(gt_order, str):
            n = len(gt_order.split())
        else:
            n = len(gt_order)

        # If a notebook has 0 or 1 cell, it contributes nothing to the swap count or normalization
        if n <= 1:
            continue

        # Calculate swaps for this notebook
        swaps = count_inversions(pred_order, gt_order)

        # Accumulate totals
        total_swaps += swaps
        total_possible_pairs += n * (n - 1)

    # Avoid division by zero if dataset is empty or all notebooks are trivial
    if total_possible_pairs == 0:
        return 1.0

    # Compute final metric
    score = 1 - 4 * (total_swaps / total_possible_pairs)

    return score
