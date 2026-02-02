import json
import os
import re
import pandas as pd
import numpy as np


def read_notebook_json(filepath):
    """
    Reads a JSON notebook file safely.

    Args:
        filepath (str): Relative or absolute path to the JSON file.

    Returns:
        dict: The parsed JSON content, or an empty dict if error/missing.
    """
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def preprocess_text(text):
    """
    Cleans source code and markdown text.
    - Normalizes whitespace (converts newlines/tabs to spaces).
    - Strips leading/trailing whitespace.
    - Lowercases the text.
    - Preserves accents (does not perform unicode normalization) as per configuration.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Replace one or more whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)

    # Strip and lowercase
    text = text.strip().lower()

    return text


def _count_inversions_recursive(a):
    """
    Recursive helper for counting inversions using Merge Sort.

    Args:
        a (list): List of integers.

    Returns:
        tuple: (number of inversions, sorted list)
    """
    if len(a) <= 1:
        return 0, a

    mid = len(a) // 2
    left_inv, left_sorted = _count_inversions_recursive(a[:mid])
    right_inv, right_sorted = _count_inversions_recursive(a[mid:])

    merge_inv = 0
    merged = []
    i, j = 0, 0

    len_left = len(left_sorted)
    len_right = len(right_sorted)

    while i < len_left and j < len_right:
        if left_sorted[i] <= right_sorted[j]:
            merged.append(left_sorted[i])
            i += 1
        else:
            merged.append(right_sorted[j])
            j += 1
            # Inversion found: left[i] > right[j]
            # Since left is sorted, all remaining elements in left are also > right[j]
            merge_inv += len_left - i

    merged.extend(left_sorted[i:])
    merged.extend(right_sorted[j:])

    return left_inv + right_inv + merge_inv, merged


def count_inversions(a):
    """
    Counts the number of inversions in a list using an O(n log n) Merge Sort approach.

    Args:
        a (list): List of comparable elements (e.g., integers).

    Returns:
        int: Total number of inversions (swaps) needed to sort the list.
    """
    inv, _ = _count_inversions_recursive(a)
    return inv


def kendall_tau_metric(df_gt, df_pred):
    """
    Calculates the Kendall Tau correlation metric as defined in the competition.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_gt (pd.DataFrame): Ground truth dataframe with columns ['id', 'cell_order'].
                              'cell_order' is a space-delimited string of cell IDs.
        df_pred (pd.DataFrame): Predictions dataframe with columns ['id', 'cell_order'].

    Returns:
        float: The Kendall Tau score.
    """
    # Create dictionaries for fast lookup
    gt_dict = dict(zip(df_gt["id"], df_gt["cell_order"]))
    pred_dict = dict(zip(df_pred["id"], df_pred["cell_order"]))

    total_swaps = 0
    total_possible = 0

    # Intersection of IDs to ensure we only score valid entries present in both
    common_ids = set(gt_dict.keys()).intersection(set(pred_dict.keys()))

    for nb_id in common_ids:
        gt_order_str = gt_dict[nb_id]
        pred_order_str = pred_dict[nb_id]

        if not isinstance(gt_order_str, str) or not isinstance(pred_order_str, str):
            continue

        gt_order = gt_order_str.split()
        pred_order = pred_order_str.split()

        n = len(gt_order)
        if n <= 1:
            continue

        # Map cell IDs to their ground truth rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert prediction to ranks, filtering only cells that exist in ground truth
        # This handles cases where predictions might have missing or extra cells gracefully
        pred_ranks = [
            rank_map[cell_id] for cell_id in pred_order if cell_id in rank_map
        ]

        # Calculate swaps (inversions)
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    if total_possible == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score
