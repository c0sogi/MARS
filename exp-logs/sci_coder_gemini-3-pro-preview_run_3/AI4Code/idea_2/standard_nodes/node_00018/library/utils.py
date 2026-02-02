import sys
import os
from bisect import bisect_left
from typing import List, Union, Dict

# Import Config to reuse the centralized seed setting logic
from library.config import Config


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across random, numpy, and torch
    by delegating to the Config class.

    Args:
        seed (int): The seed value to use.
    """
    Config.set_seed(seed)


def count_inversions(a: List[int]) -> int:
    """
    Counts the number of inversions in a list using a bisect-based approach.
    This calculates how many swaps are needed to sort the array (bubble sort distance).

    Args:
        a (List[int]): A list of integers (ranks).

    Returns:
        int: The number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect_left finds the first index where x can be inserted while maintaining order.
        # Elements in sorted_so_far at indices >= idx are strictly greater than x (if duplicates handled)
        # or generally, these are the elements that appeared before x but are larger than x.
        idx = bisect_left(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(
    predictions: Union[List[List[str]], Dict[str, List[str]]],
    ground_truths: Union[List[List[str]], Dict[str, List[str]]],
) -> float:
    """
    Computes the Kendall Tau correlation metric as defined in the competition task.

    The formula used is:
    K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))

    Where:
    - S_i is the number of swaps (inversions) needed to sort the predicted order into the ground truth.
    - n_i is the number of cells in the notebook.

    Args:
        predictions: A list of predicted cell orderings (lists of IDs) or a dictionary mapping
                     notebook IDs to their predicted cell orderings.
        ground_truths: A list of ground truth cell orderings (lists of IDs) or a dictionary mapping
                       notebook IDs to their correct cell orderings.

    Returns:
        float: The accumulated Kendall Tau correlation score across all samples.
    """

    # Normalize inputs to lists of lists if they are dictionaries
    if isinstance(predictions, dict) and isinstance(ground_truths, dict):
        # Align by common keys to ensure correct comparison
        common_keys = sorted(list(set(predictions.keys()) & set(ground_truths.keys())))
        preds_list = [predictions[k] for k in common_keys]
        truth_list = [ground_truths[k] for k in common_keys]
    else:
        preds_list = predictions
        truth_list = ground_truths

    total_inversions = 0
    total_normalization_term = 0  # This corresponds to Sum(n_i * (n_i - 1))

    for pred, true in zip(preds_list, truth_list):
        n = len(true)
        # If a notebook has 0 or 1 cell, ordering is trivial and contributes nothing to the denominator
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1) in the ground truth
        true_rank_map = {cell_id: i for i, cell_id in enumerate(true)}

        # Convert the predicted cell sequence into a sequence of ranks based on the ground truth
        # We filter to ensure we only consider cells present in the ground truth
        pred_ranks = []
        for cell_id in pred:
            if cell_id in true_rank_map:
                pred_ranks.append(true_rank_map[cell_id])

        # Calculate inversions (S_i) for this notebook
        s_i = count_inversions(pred_ranks)

        total_inversions += s_i
        total_normalization_term += n * (n - 1)

    # Avoid division by zero if the dataset is empty or contains only trivial notebooks
    if total_normalization_term == 0:
        return 1.0

    score = 1.0 - 4.0 * (total_inversions / total_normalization_term)
    return score
