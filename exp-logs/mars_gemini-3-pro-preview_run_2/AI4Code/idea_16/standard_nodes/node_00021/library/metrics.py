import pandas as pd
from bisect import bisect
from typing import Dict, List, Union
from library.config import Config


def count_inversions(a: List[int]) -> int:
    """
    Counts the number of inversions in a list of integers using a bisect approach.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    This represents the minimum number of swaps needed to sort the array.

    Args:
        a (List[int]): A list of integers (ranks).

    Returns:
        int: The number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the insertion point to maintain sorted order
        idx = bisect(sorted_so_far, x)
        # All elements already in sorted_so_far to the right of idx are larger than x
        # and appeared earlier in the sequence, thus forming inversions.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(ground_truth: List[str], prediction: List[str]) -> int:
    """
    Computes the number of swaps (inversions) needed to transform the prediction
    sequence into the ground truth sequence.

    Args:
        ground_truth (List[str]): List of cell_ids in the correct order.
        prediction (List[str]): List of cell_ids in the predicted order.

    Returns:
        int: The number of swaps S required.
    """
    # Map each cell_id in ground_truth to its correct rank (0, 1, 2, ...)
    rank_map = {cell_id: i for i, cell_id in enumerate(ground_truth)}

    # Convert the predicted sequence of cell IDs into a sequence of ranks.
    # We filter out any IDs in prediction that aren't in ground_truth to be robust,
    # though valid submissions should contain the exact same set of IDs.
    predicted_ranks = [
        rank_map[cell_id] for cell_id in prediction if cell_id in rank_map
    ]

    return count_inversions(predicted_ranks)


def compute_score(
    df_ground_truth: pd.DataFrame, predictions: Dict[str, Union[List[str], str]]
) -> float:
    """
    Computes the global Kendall tau correlation metric aggregated across the dataset.

    Formula: K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))

    Args:
        df_ground_truth (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns.
                                        'cell_order' is a space-delimited string of cell IDs.
        predictions (Dict): Dictionary mapping notebook 'id' to the predicted order.
                            Values can be a list of strings or a space-delimited string.

    Returns:
        float: The calculated Kendall tau score.
    """
    total_swaps = 0
    total_pairs = 0

    # Iterate over each notebook in the ground truth dataframe
    for _, row in df_ground_truth.iterrows():
        nb_id = row["id"]
        gt_order_str = row["cell_order"]

        # Skip if no prediction exists for this ID
        if nb_id not in predictions:
            continue

        # Parse ground truth
        gt_order = gt_order_str.split()
        n = len(gt_order)

        # If a notebook has 0 or 1 cell, it contributes nothing to the denominator n(n-1)
        if n <= 1:
            continue

        # Parse prediction
        pred_input = predictions[nb_id]
        if isinstance(pred_input, str):
            pred_order = pred_input.split()
        else:
            pred_order = pred_input

        # Calculate swaps (S_i) for this notebook
        swaps = compute_kendall_tau(gt_order, pred_order)

        # Accumulate totals
        total_swaps += swaps
        total_pairs += n * (n - 1)

    # Avoid division by zero
    if total_pairs == 0:
        return 0.0

    # Calculate final metric
    score = 1 - 4 * (total_swaps / total_pairs)
    return score
