import os
import random
import numpy as np
import pandas as pd
import sys


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # robustly set torch seed if available, without crashing if not
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def count_inversions(a):
    """
    Counts the number of inversions in a list using Merge Sort.
    Time Complexity: O(N log N)
    """
    inversions = 0
    if len(a) <= 1:
        return 0, a

    mid = len(a) // 2
    left_inv, left_sorted = count_inversions(a[:mid])
    right_inv, right_sorted = count_inversions(a[mid:])

    merge_inv = 0
    merged = []
    i, j = 0, 0

    while i < len(left_sorted) and j < len(right_sorted):
        if left_sorted[i] <= right_sorted[j]:
            merged.append(left_sorted[i])
            i += 1
        else:
            merged.append(right_sorted[j])
            j += 1
            # If left[i] > right[j], then left[i] and all subsequent elements in left
            # are inversions with respect to right[j].
            merge_inv += len(left_sorted) - i

    merged += left_sorted[i:]
    merged += right_sorted[j:]

    return left_inv + right_inv + merge_inv, merged


def compute_kendall_tau(df_ground_truth, df_predictions):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_ground_truth (pd.DataFrame): DataFrame with 'id' and 'cell_order' columns.
        df_predictions (pd.DataFrame): DataFrame with 'id' and 'cell_order' columns.

    Returns:
        float: The Kendall Tau score.
    """
    # Create dictionaries for fast lookup
    gt_dict = dict(zip(df_ground_truth["id"], df_ground_truth["cell_order"]))
    pred_dict = dict(zip(df_predictions["id"], df_predictions["cell_order"]))

    total_swaps = 0
    total_denominator = 0

    # Intersection of IDs ensures we only score valid entries
    common_ids = set(gt_dict.keys()).intersection(set(pred_dict.keys()))

    for nb_id in common_ids:
        gt_order = gt_dict[nb_id].split()
        pred_order = pred_dict[nb_id].split()

        n = len(gt_order)
        if n <= 1:
            continue

        # Map cell IDs to their ground truth rank (0 to n-1)
        # If a cell ID in pred is not in gt (should not happen in valid submission), ignore or handle
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert prediction to a list of ranks based on ground truth
        # We filter to ensure we only consider cells present in both (safety check)
        pred_ranks = [
            gt_rank_map[cell_id] for cell_id in pred_order if cell_id in gt_rank_map
        ]

        # If lengths mismatch significantly, it indicates an issue, but we proceed with what we have
        current_n = len(pred_ranks)

        # Calculate swaps (inversions) needed to sort pred_ranks
        swaps, _ = count_inversions(pred_ranks)

        total_swaps += swaps
        total_denominator += n * (n - 1)

    if total_denominator == 0:
        return 0.0

    kendall_tau = 1 - 4 * (total_swaps / total_denominator)

    # Print full precision as requested
    print(f"Validation Kendall Tau: {kendall_tau}")

    return kendall_tau


def format_submission(ids, cell_orders):
    """
    Formats the predictions into a DataFrame suitable for submission.

    Args:
        ids (list): List of notebook IDs.
        cell_orders (list): List of space-delimited strings representing cell order.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'cell_order'].
    """
    df_submission = pd.DataFrame({"id": ids, "cell_order": cell_orders})
    return df_submission
