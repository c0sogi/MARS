import json
import os
import torch
import pandas as pd
from library.config import Config, set_seed


def get_device():
    """
    Returns the PyTorch device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(filepath):
    """
    Reads a notebook JSON file and returns the data as a dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: The parsed JSON content.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def count_inversions(arr):
    """
    Counts the number of inversions in a list.
    An inversion is a pair (i, j) such that i < j and arr[i] > arr[j].

    Args:
        arr (list): List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    n = len(arr)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count


def compute_kendall_tau(df_preds, df_gt):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_preds (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).

    Returns:
        float: The Kendall Tau score.
    """
    # Merge predictions with ground truth on 'id'
    df_merged = pd.merge(
        df_gt[["id", "cell_order"]],
        df_preds[["id", "cell_order"]],
        on="id",
        suffixes=("_gt", "_pred"),
    )

    total_swaps = 0
    total_possible_pairs = 0

    for _, row in df_merged.iterrows():
        gt_order = row["cell_order_gt"].split()
        pred_order = row["cell_order_pred"].split()

        n = len(gt_order)
        if n <= 1:
            continue

        # Map cell IDs to their correct rank (index) in the ground truth
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert predicted order to a list of ranks based on ground truth
        # We filter to ensure we only consider valid cells present in the ground truth
        pred_ranks = []
        for cell_id in pred_order:
            if cell_id in gt_rank_map:
                pred_ranks.append(gt_rank_map[cell_id])

        # Calculate swaps (inversions) needed to sort pred_ranks
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible_pairs += n * (n - 1)

    if total_possible_pairs == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible_pairs)
    return score


def get_ranks(pred_scores, code_cells):
    """
    Converts soft predictions into a final cell ordering by interleaving markdown cells
    into the fixed sequence of code cells based on predicted scores.

    Args:
        pred_scores (dict): Dictionary mapping markdown cell_id to predicted score (float).
                            Lower score = earlier position.
        code_cells (list): List of code cell IDs in their correct, fixed order.

    Returns:
        str: Space-delimited string of the sorted cell order.
    """
    cells_with_scores = []

    # Assign integer scores to code cells to anchor them at 0.0, 1.0, 2.0, etc.
    for i, cell_id in enumerate(code_cells):
        cells_with_scores.append((cell_id, float(i)))

    # Add markdown cells with their predicted float scores
    # e.g., a score of 0.5 places the markdown cell between code cell 0 and 1
    for cell_id, score in pred_scores.items():
        cells_with_scores.append((cell_id, float(score)))

    # Sort all cells by score
    cells_with_scores.sort(key=lambda x: x[1])

    # Extract the sorted cell IDs
    sorted_ids = [x[0] for x in cells_with_scores]

    return " ".join(sorted_ids)
