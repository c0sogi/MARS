import pandas as pd
from bisect import bisect
from library.config import *


def count_inversions(a):
    """
    Counts the number of inversions in a list using bisect.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    This is equivalent to the number of swaps needed to sort the array.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect finds the insertion point to maintain sorted order.
        # Elements to the right of this point in 'sorted_so_far' are larger than x
        # and appeared earlier in the sequence 'a', thus forming inversions with x.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def calculate_kendall_tau(predictions, ground_truths):
    """
    Calculates the Kendall tau correlation metric as defined in the task.

    Formula: K = 1 - 4 * (Sum S_i) / (Sum n_i * (n_i - 1))
    Where S_i is the number of swaps (inversions) for notebook i,
    and n_i is the number of cells in notebook i.

    Args:
        predictions (dict): Map from notebook_id to list of predicted cell_ids.
        ground_truths (dict): Map from notebook_id to list of ground truth cell_ids.

    Returns:
        float: The Kendall tau score.
    """
    total_inversions = 0
    total_max_inversions = 0

    # Iterate over all notebooks present in the ground truth
    for nb_id, gt_order in ground_truths.items():
        # Skip if no prediction is provided for this notebook
        if nb_id not in predictions:
            continue

        pred_order = predictions[nb_id]
        n = len(gt_order)

        # If a notebook has 0 or 1 cell, the denominator n*(n-1) is 0.
        # These cases do not contribute to the metric.
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert the predicted cell order into a list of ranks based on ground truth.
        # We filter for valid keys to handle potential discrepancies, though
        # predictions are expected to be permutations of the ground truth.
        pred_ranks = []
        for cell_id in pred_order:
            if cell_id in gt_rank_map:
                pred_ranks.append(gt_rank_map[cell_id])

        # Count inversions (swaps needed to sort the predicted ranks into [0, 1, ...])
        s = count_inversions(pred_ranks)

        total_inversions += s
        total_max_inversions += n * (n - 1)

    # Avoid division by zero if the dataset is empty or contains only trivial notebooks
    if total_max_inversions == 0:
        return 1.0

    # Compute final metric
    score = 1 - 4 * (total_inversions / total_max_inversions)
    return score


def score_dataset(df_val, df_pred):
    """
    Computes the metric for a validation dataset provided as DataFrames.

    Args:
        df_val (pd.DataFrame): Validation metadata containing 'id' and 'cell_order'.
        df_pred (pd.DataFrame): Predictions containing 'id' and 'cell_order'.

    Returns:
        float: The Kendall tau score.
    """

    # Helper function to ensure cell_order is a list of strings
    def to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return x.split()
        return []

    # Convert DataFrames to dictionaries for efficient lookup
    # Keys are notebook IDs, values are lists of cell IDs
    gt_dict = dict(zip(df_val["id"], df_val["cell_order"].apply(to_list)))
    pred_dict = dict(zip(df_pred["id"], df_pred["cell_order"].apply(to_list)))

    score = calculate_kendall_tau(pred_dict, gt_dict)
    return score
