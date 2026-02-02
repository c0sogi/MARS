import json
import bisect
import os
import pandas as pd
import numpy as np
from library.config import Config


def read_notebook_json(filepath):
    """
    Reads a notebook JSON file safely.

    Args:
        filepath (str): Full path to the JSON file.

    Returns:
        dict: A dictionary containing 'cell_type' and 'source' dictionaries.
              Returns None if the file cannot be read.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading notebook file {filepath}: {e}")
        return None


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers using a bisect approach.
    An inversion is a pair of indices (i, j) such that i < j and a[i] > a[j].

    Args:
        a (list): List of integers.

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect_right returns the insertion point i such that all elements
        # to the left are <= x and all to the right are > x.
        # Since we are building sorted_so_far incrementally, elements currently
        # in sorted_so_far are the ones that appeared *before* x in the original list.
        # Elements at indices >= idx are greater than x, thus forming inversions.
        idx = bisect.bisect_right(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau_metric(df_gt, df_pred):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (ground truth).
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (predictions).

    Returns:
        float: The Kendall Tau score.
    """
    # Ensure we are working with string types for IDs
    df_gt = df_gt.copy()
    df_pred = df_pred.copy()
    df_gt["id"] = df_gt["id"].astype(str)
    df_pred["id"] = df_pred["id"].astype(str)

    # Merge predictions with ground truth on notebook ID
    df = df_gt.merge(df_pred, on="id", suffixes=("_gt", "_pred"))

    total_swaps = 0
    total_possible_swaps = 0

    for _, row in df.iterrows():
        gt_order = row["cell_order_gt"].split()
        pred_order = row["cell_order_pred"].split()

        n = len(gt_order)
        # If a notebook has 0 or 1 cell, no swaps are possible/needed
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0-indexed position)
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Translate the predicted order into a list of ground truth ranks.
        # We filter to ensure we only consider cells present in the ground truth
        # (ignoring potential hallucinations, though robust predictions shouldn't have them).
        pred_indices = []
        for cell_id in pred_order:
            if cell_id in gt_rank_map:
                pred_indices.append(gt_rank_map[cell_id])

        # The number of swaps to sort pred_indices is equal to the number of inversions
        swaps = count_inversions(pred_indices)

        total_swaps += swaps
        total_possible_swaps += n * (n - 1)

    if total_possible_swaps == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible_swaps)
    return score


def convert_ranks_to_order(markdown_ranks, code_cells):
    """
    Converts predicted ranks for markdown cells and a fixed list of code cells
    into a final ordered list of cell IDs.

    Strategy:
    - Code cells are assigned fixed, equidistant ranks in the interval [0.0, 1.0].
    - Markdown cells use their predicted continuous ranks.
    - All cells are combined and sorted by rank.

    Args:
        markdown_ranks (dict): Dictionary mapping {cell_id: predicted_rank}.
        code_cells (list): List of code cell IDs in their correct relative order.

    Returns:
        str: Space-delimited string of ordered cell IDs.
    """
    cells_with_ranks = []

    # Add markdown cells with their predicted ranks
    for cell_id, rank in markdown_ranks.items():
        cells_with_ranks.append((cell_id, rank))

    # Add code cells with fixed equidistant ranks
    n_code = len(code_cells)
    if n_code > 0:
        if n_code == 1:
            # If there is only one code cell, we place it at 0.0
            code_ranks = [0.0]
        else:
            # Linspace from 0.0 to 1.0
            code_ranks = np.linspace(0.0, 1.0, n_code)

        for cell_id, rank in zip(code_cells, code_ranks):
            cells_with_ranks.append((cell_id, rank))

    # Sort all cells by rank (ascending)
    # Python's sort is stable, which is good for tie-breaking if necessary
    cells_with_ranks.sort(key=lambda x: x[1])

    # Extract just the IDs
    ordered_ids = [x[0] for x in cells_with_ranks]

    return " ".join(ordered_ids)
