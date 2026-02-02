import os
import json
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_notebook(filepath):
    """
    Reads a notebook JSON file and extracts cell data.

    Args:
        filepath (str): Path to the .json notebook file.

    Returns:
        tuple: (code_ids, markdown_ids, source_dict)
            - code_ids (list): List of code cell IDs in their original order.
            - markdown_ids (list): List of markdown cell IDs (shuffled).
            - source_dict (dict): Dictionary mapping cell_id to source text.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Notebook file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
        return [], [], {}

    cell_type = data.get("cell_type", {})
    source = data.get("source", {})

    # In Python 3.7+, dictionary insertion order is preserved.
    # For the training set, code cells are in the correct order,
    # and markdown cells are appended in a shuffled order.
    # For the test set, code cells are also in the correct relative order.
    code_ids = [cid for cid, ctype in cell_type.items() if ctype == "code"]
    markdown_ids = [cid for cid, ctype in cell_type.items() if ctype == "markdown"]

    return code_ids, markdown_ids, source


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].

    Args:
        a (list): List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    n = len(a)
    # Using simple O(N^2) approach as N is typically small (< 500)
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] > a[j]:
                inversions += 1
    return inversions


def compute_kendall_tau(predictions, ground_truths):
    """
    Computes the accumulated Kendall Tau correlation metric across a collection of notebooks.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        predictions (list of list of str): Predicted cell orders (list of cell IDs).
        ground_truths (list of list of str): Ground truth cell orders (list of cell IDs).

    Returns:
        float: The Kendall Tau score.
    """
    total_swaps = 0
    total_pairs = 0

    for pred, true in zip(predictions, ground_truths):
        n = len(true)
        if n < 2:
            continue

        # Map true cell IDs to their correct rank (0 to n-1)
        true_rank = {cid: i for i, cid in enumerate(true)}

        # Convert prediction sequence to a list of ranks based on ground truth.
        # We filter to ensure we only consider IDs present in the ground truth
        # (handling potential mismatches robustly).
        pred_ranks = [true_rank[cid] for cid in pred if cid in true_rank]

        # Calculate number of swaps (inversions) needed to sort pred into true
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_pairs += n * (n - 1)

    if total_pairs == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_pairs)
    return score


def get_ordered_cell_ids(code_ids, markdown_ids, markdown_scores):
    """
    Constructs the final ordered list of cell IDs by interleaving code and markdown cells.

    Args:
        code_ids (list): List of code cell IDs (anchors).
        markdown_ids (list): List of markdown cell IDs.
        markdown_scores (list): Predicted continuous positions for markdown cells.
                                A score of X implies the cell is positioned near code cell index X.

    Returns:
        str: Space-delimited string of the ordered cell IDs.
    """
    cells = []

    # Assign fixed positions to code cells.
    # Code cell i is placed at position i + 0.5.
    # This acts as a boundary: if a markdown cell has score < 0.5, it goes before code cell 0.
    # If score is between 0.5 and 1.5, it goes between code cell 0 and 1.
    for i, cid in enumerate(code_ids):
        cells.append((i + 0.5, cid))

    # Add markdown cells with their predicted scores
    for cid, score in zip(markdown_ids, markdown_scores):
        cells.append((score, cid))

    # Sort all cells by their position score
    cells.sort(key=lambda x: x[0])

    # Extract IDs in order
    ordered_ids = [cid for _, cid in cells]

    return " ".join(ordered_ids)
