from bisect import bisect
from typing import List, Union, Dict
import pandas as pd


def count_inversions(a: List[int]) -> int:
    """
    Counts the number of inversions in a list.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    This corresponds to the number of swaps needed to sort the array into ascending order.

    Args:
        a: List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to maintain order.
        # Elements in sorted_so_far at indices >= idx are greater than x.
        # Since they appeared earlier in the sequence 'a', they form inversions with x.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau(
    ground_truth: List[Union[str, List[str]]], predictions: List[Union[str, List[str]]]
) -> float:
    """
    Computes the Kendall Tau correlation metric as defined for the competition.

    Formula: K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))
    Where S_i is the number of swaps needed to sort the predicted order to the ground truth.

    Args:
        ground_truth: List of correct cell orders. Each element can be a space-delimited string or a list of cell IDs.
        predictions: List of predicted cell orders. Each element can be a space-delimited string or a list of cell IDs.

    Returns:
        float: The Kendall Tau score.
    """
    total_swaps = 0
    total_possible = 0

    for gt, pred in zip(ground_truth, predictions):
        # Normalize inputs to lists of strings
        if isinstance(gt, str):
            gt = gt.split()
        if isinstance(pred, str):
            pred = pred.split()

        n = len(gt)
        # Notebooks with 0 or 1 cell do not contribute to the denominator or swaps
        if n <= 1:
            continue

        # Map ground truth cell IDs to their correct rank (0, 1, ..., n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(gt)}

        # Convert predicted sequence to a list of ranks based on ground truth
        # We filter to ensure we only consider cells present in the ground truth
        pred_ranks = []
        for cell_id in pred:
            if cell_id in rank_map:
                pred_ranks.append(rank_map[cell_id])

        # Calculate the number of swaps (inversions) needed to sort the predictions
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    # Avoid division by zero if all notebooks are trivial
    if total_possible == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score


def compute_score(
    df_val: pd.DataFrame, predictions: Union[Dict, pd.Series, List]
) -> float:
    """
    Wrapper function to compute the metric using a validation DataFrame.

    Args:
        df_val: Validation DataFrame containing at least 'id' and 'cell_order' columns.
        predictions: The predicted orders. Can be:
            - A dictionary mapping notebook 'id' to predicted 'cell_order' (str or list).
            - A pandas Series or list of predicted orders aligned with df_val.

    Returns:
        float: The Kendall Tau score.
    """
    ground_truth = df_val["cell_order"].tolist()

    preds_list = []

    if isinstance(predictions, dict):
        # If predictions are a dict, map them using the IDs in the validation set
        for nb_id in df_val["id"]:
            # Default to empty list if ID not found (though usually should be present)
            preds_list.append(predictions.get(nb_id, ""))
    elif isinstance(predictions, pd.Series):
        preds_list = predictions.tolist()
    else:
        # Assume list is already aligned
        preds_list = predictions

    return kendall_tau(ground_truth, preds_list)
