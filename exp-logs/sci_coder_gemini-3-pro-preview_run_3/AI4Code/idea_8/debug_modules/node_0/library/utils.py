import os
import json
import random
import numpy as np
import torch
import pandas as pd
from bisect import bisect, insort
from library.config import Config


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_notebook(file_path):
    """
    Reads a notebook JSON file and returns its cell_type and source dictionaries.

    Args:
        file_path (str): Path to the .json notebook file.

    Returns:
        tuple: (cell_type_dict, source_dict) or (None, None) if error.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cell_type", {}), data.get("source", {})
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}, {}


def preprocess_text(text):
    """
    Basic text cleaning: lowercase and strip whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)
    return text.lower().strip()


def count_inversions(a):
    """
    Counts the number of inversions in a list using bisect.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    This is equivalent to the number of swaps needed to sort the list.

    Args:
        a (list): List of comparable elements (e.g., integers).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect returns the insertion point to maintain sorted order
        # elements to the right of this point in sorted_so_far are greater than x
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        insort(sorted_so_far, x)
    return inversions


def calc_kendall_tau(df_ground_truth, submission_dict):
    """
    Computes the global Kendall Tau correlation metric.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_ground_truth (pd.DataFrame): DataFrame with 'id' and 'cell_order' columns.
        submission_dict (dict): Dictionary mapping notebook 'id' to predicted 'cell_order'
                                (list of strings or space-delimited string).

    Returns:
        float: The calculated metric.
    """
    total_swaps = 0
    total_pairs = 0

    # Iterate through ground truth
    for _, row in df_ground_truth.iterrows():
        nb_id = row["id"]
        gt_order = row["cell_order"].split()

        if nb_id not in submission_dict:
            # If prediction is missing, worst case assumes reverse order?
            # Or simply skip. Usually submission must be complete.
            # We will assume worst case (max swaps) for missing predictions
            # to penalize heavily, or 0 contribution to score if strictly checking.
            # Given the formula, missing predictions would break the logic.
            # We assume submission_dict is complete.
            continue

        pred_order = submission_dict[nb_id]
        if isinstance(pred_order, str):
            pred_order = pred_order.split()

        # Validate lengths
        n = len(gt_order)
        if len(pred_order) != n:
            print(
                f"Warning: Length mismatch for {nb_id}. GT: {n}, Pred: {len(pred_order)}"
            )
            # Fallback: Treat as worst case for this notebook
            total_swaps += n * (n - 1) // 2
            total_pairs += n * (n - 1)
            continue

        # Map cell IDs to their ground truth rank
        # gt_order = ['a', 'b', 'c'] -> rank_map = {'a': 0, 'b': 1, 'c': 2}
        rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert predicted order to a list of ground truth ranks
        # If pred_order = ['b', 'a', 'c'], ranks = [1, 0, 2]
        # Inversions in [1, 0, 2] = 1 (1 > 0)
        try:
            pred_ranks = [rank_map[cell_id] for cell_id in pred_order]
        except KeyError:
            # Prediction contains ID not in GT
            print(f"Warning: Unknown cell ID in prediction for {nb_id}")
            total_swaps += n * (n - 1) // 2
            total_pairs += n * (n - 1)
            continue

        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_pairs += n * (n - 1)

    if total_pairs == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_pairs)
    return score


def reconstruct_order(code_cells, markdown_scores):
    """
    Reconstructs the final cell order by merging fixed code cells and ranked markdown cells.

    Args:
        code_cells (list): List of code cell IDs in their correct relative order.
        markdown_scores (dict): Dictionary mapping markdown cell IDs to their predicted rank/score.
                                Lower score/rank implies earlier position.

    Returns:
        str: Space-delimited string of the final cell order.
    """
    # Assign code cells integer positions: 0, 1, 2, ...
    # This assumes markdown scores are scaled to be comparable to these indices.
    # If the model predicts normalized rank [0, 1], the caller should have scaled
    # them by len(code_cells) or we treat code cells as anchors.
    # Here we assume markdown_scores are directly comparable to code indices.

    cells = []

    # Add code cells with integer ranks
    for i, cell_id in enumerate(code_cells):
        cells.append((cell_id, float(i)))

    # Add markdown cells with predicted ranks
    for cell_id, score in markdown_scores.items():
        cells.append((cell_id, float(score)))

    # Sort by score
    # Stable sort ensures that if a markdown cell has exact same score as a code cell,
    # order is preserved (though floats usually differ).
    cells.sort(key=lambda x: x[1])

    # Extract IDs
    final_order = [c[0] for c in cells]

    return " ".join(final_order)
