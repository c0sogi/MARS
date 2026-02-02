import os
import json
import pandas as pd
from bisect import bisect
from library.config import Config


def read_notebook(filepath):
    """
    Reads a JSON notebook file and returns its content.

    Args:
        filepath (str): Full path to the .json file.

    Returns:
        dict: A dictionary containing 'cell_type' and 'source' keys.
              Returns an empty dict if reading fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading notebook at {filepath}: {e}")
        return {}


def count_inversions(arr):
    """
    Counts the number of inversions in a list.
    This corresponds to the number of swaps needed to sort the array
    into ascending order (0, 1, 2, ...).

    Args:
        arr (list): List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_list = []
    for x in arr:
        # Find the position where x should be inserted to keep the list sorted
        idx = bisect(sorted_list, x)
        # The number of elements strictly greater than x that have already been seen
        # is equal to the number of elements currently in sorted_list after index idx.
        inversions += len(sorted_list) - idx
        sorted_list.insert(idx, x)
    return inversions


def kendall_tau(ground_truth, predictions):
    """
    Computes the Kendall Tau correlation metric as defined for the task.
    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        ground_truth (pd.DataFrame): DataFrame with columns ['id', 'cell_order'].
        predictions (pd.DataFrame): DataFrame with columns ['id', 'cell_order'].

    Returns:
        float: The calculated metric score.
    """
    # Convert DataFrames to dictionaries for O(1) lookup
    gt_map = dict(zip(ground_truth["id"], ground_truth["cell_order"]))
    pred_map = dict(zip(predictions["id"], predictions["cell_order"]))

    total_swaps = 0
    total_denom = 0

    # Process only notebooks present in both sets (intersection)
    common_ids = set(gt_map.keys()) & set(pred_map.keys())

    for nb_id in common_ids:
        gt_order = gt_map[nb_id].split()
        pred_order = pred_map[nb_id].split()

        n = len(gt_order)
        # A notebook with 0 or 1 cell has 0 possible swaps, contributing nothing to denominator
        if n <= 1:
            continue

        # Map cell_id to its correct rank (0 to n-1)
        rank_lookup = {cell_id: r for r, cell_id in enumerate(gt_order)}

        # Convert the predicted cell order into a list of ranks based on ground truth
        # We filter to ensure we only process cells that exist in the ground truth
        pred_ranks = [rank_lookup[cid] for cid in pred_order if cid in rank_lookup]

        # Count inversions (swaps)
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_denom += n * (n - 1)

    if total_denom == 0:
        return 0.0

    return 1 - 4 * (total_swaps / total_denom)


def format_submission(ids, cell_orders):
    """
    Formats predictions into a DataFrame for submission.

    Args:
        ids (list): List of notebook IDs.
        cell_orders (list): List of cell orders. Each element can be a
                            list of cell IDs or a space-delimited string.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'cell_order'].
    """
    formatted_orders = []
    for order in cell_orders:
        if isinstance(order, list):
            formatted_orders.append(" ".join(order))
        else:
            formatted_orders.append(str(order))

    return pd.DataFrame({"id": ids, "cell_order": formatted_orders})
