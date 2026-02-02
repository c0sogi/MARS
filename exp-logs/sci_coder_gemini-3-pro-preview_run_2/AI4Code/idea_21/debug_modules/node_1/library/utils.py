import json
import os
import numpy as np
import pandas as pd
from library.config import Config


def read_json_file(filepath):
    """
    Safely loads a JSON file from the disk.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: The loaded JSON content, or an empty dict if loading fails.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}


def count_inversions(arr):
    """
    Counts the number of inversions in a list using a Merge Sort based approach.
    An inversion is a pair (i, j) such that i < j and arr[i] > arr[j].

    Time Complexity: O(N log N)

    Args:
        arr (list): List of comparable elements (e.g., integers representing ranks).

    Returns:
        tuple: (number_of_inversions, sorted_list)
    """
    if len(arr) <= 1:
        return 0, arr

    mid = len(arr) // 2
    left_inv, left_sorted = count_inversions(arr[:mid])
    right_inv, right_sorted = count_inversions(arr[mid:])

    merge_inv = 0
    merged = []
    i, j = 0, 0

    # Merge step
    while i < len(left_sorted) and j < len(right_sorted):
        if left_sorted[i] <= right_sorted[j]:
            merged.append(left_sorted[i])
            i += 1
        else:
            merged.append(right_sorted[j])
            j += 1
            # All remaining elements in left_sorted are greater than right_sorted[j]
            merge_inv += len(left_sorted) - i

    merged.extend(left_sorted[i:])
    merged.extend(right_sorted[j:])

    return left_inv + right_inv + merge_inv, merged


def kendall_tau_metric(df_true, df_pred):
    """
    Computes the Kendall Tau correlation metric as defined for the AI4Code competition.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n * (n - 1))

    Args:
        df_true (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (ground truth).
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (predictions).

    Returns:
        float: The calculated Kendall Tau score.
    """
    # Merge to ensure alignment of notebooks
    df = df_true[["id", "cell_order"]].merge(
        df_pred[["id", "cell_order"]], on="id", suffixes=("_true", "_pred")
    )

    total_swaps = 0
    total_denom = 0

    for _, row in df.iterrows():
        true_order_str = row["cell_order_true"]
        pred_order_str = row["cell_order_pred"]

        if not isinstance(true_order_str, str) or not isinstance(pred_order_str, str):
            continue

        true_order = true_order_str.split()
        pred_order = pred_order_str.split()

        n = len(true_order)
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(true_order)}

        # Convert the predicted order into a list of ranks based on ground truth
        # Filter out invalid IDs if any
        pred_ranks = []
        for cell_id in pred_order:
            if cell_id in rank_map:
                pred_ranks.append(rank_map[cell_id])

        # Calculate swaps (inversions) needed to sort the predicted ranks
        swaps, _ = count_inversions(pred_ranks)

        total_swaps += swaps
        total_denom += n * (n - 1)

    if total_denom == 0:
        return 0.0

    return 1 - 4 * (total_swaps / total_denom)


def convert_ranks_to_order(code_cells, markdown_cells, markdown_ranks):
    """
    Generates the final cell order string by combining fixed code cells and
    ranked markdown cells.

    Code cells are assigned equidistant ranks in [0, 1]. Markdown cells are
    placed based on their predicted ranks.

    Args:
        code_cells (list): List of code cell IDs (strings).
        markdown_cells (list): List of markdown cell IDs (strings).
        markdown_ranks (list or np.array): Predicted ranks for markdown cells.

    Returns:
        str: Space-delimited string of the ordered cell IDs.
    """
    if not code_cells:
        # If no code cells, sort markdown cells purely by rank
        if not markdown_cells:
            return ""
        sorted_md = sorted(zip(markdown_ranks, markdown_cells), key=lambda x: x[0])
        return " ".join([x[1] for x in sorted_md])

    if not markdown_cells:
        # If no markdown cells, return code cells as is
        return " ".join(code_cells)

    # Generate ranks for code cells
    # We use linspace to spread them evenly from 0.0 to 1.0
    n_code = len(code_cells)
    if n_code == 1:
        # Single code cell anchor at 0.0
        code_ranks = [0.0]
    else:
        code_ranks = np.linspace(0, 1, n_code)

    # Combine all cells with their ranks
    all_cells = []

    for r, cid in zip(code_ranks, code_cells):
        all_cells.append((r, cid))

    for r, cid in zip(markdown_ranks, markdown_cells):
        all_cells.append((r, cid))

    # Sort by rank
    # Python's sort is stable, but float comparison is sufficient here
    all_cells.sort(key=lambda x: x[0])

    # Extract IDs
    ordered_ids = [x[1] for x in all_cells]

    return " ".join(ordered_ids)
