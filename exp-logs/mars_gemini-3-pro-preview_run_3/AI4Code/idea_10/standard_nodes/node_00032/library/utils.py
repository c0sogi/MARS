import os
import json
import random
import numpy as np
import pandas as pd
from bisect import bisect
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_notebook(file_path):
    """
    Reads a JSON notebook file and returns the code and markdown cells.

    Args:
        file_path (str): Path to the .json notebook file.

    Returns:
        tuple: (code_cells, markdown_cells)
            - code_cells (list of dict): [{'id': cell_id, 'source': text}, ...]
            - markdown_cells (list of dict): [{'id': cell_id, 'source': text}, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    code_cells = []
    markdown_cells = []

    for cell_id, cell_type in cell_types.items():
        source = sources.get(cell_id, "")
        cell_data = {"id": cell_id, "source": source}

        if cell_type == "code":
            code_cells.append(cell_data)
        elif cell_type == "markdown":
            markdown_cells.append(cell_data)

    return code_cells, markdown_cells


def count_inversions(a):
    """
    Counts the number of inversions in a list using a Bisect approach (O(N log N)).
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].

    Args:
        a (list): List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect returns the insertion point i such that all e in sorted_so_far[:i] have e <= x
        # Elements to the right of this position in sorted_so_far are strictly greater than x.
        # Since those elements appeared *before* x in the original array 'a', they form inversions.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def calc_kendall_tau(df_gt, df_pred):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n * (n - 1))

    Args:
        df_gt (pd.DataFrame): Ground truth dataframe with columns ['id', 'cell_order'].
        df_pred (pd.DataFrame): Predictions dataframe with columns ['id', 'cell_order'].

    Returns:
        float: The Kendall Tau score.
    """
    # Convert to dictionaries for faster lookup
    gt_dict = dict(zip(df_gt["id"], df_gt["cell_order"]))
    pred_dict = dict(zip(df_pred["id"], df_pred["cell_order"]))

    total_swaps = 0
    total_possible = 0

    # Iterate over notebooks present in both
    common_ids = set(gt_dict.keys()).intersection(set(pred_dict.keys()))

    for nb_id in common_ids:
        gt_order = gt_dict[nb_id].split()
        pred_order = pred_dict[nb_id].split()

        n = len(gt_order)
        if n <= 1:
            continue

        # Map cell IDs to their ground truth rank (0 to n-1)
        gt_ranks = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert predicted order to a list of ranks based on ground truth
        # Filter out any cells not in ground truth (safety check)
        pred_ranks = [
            gt_ranks[cell_id] for cell_id in pred_order if cell_id in gt_ranks
        ]

        # Calculate swaps (inversions) needed to sort pred_ranks
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    if total_possible == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score


def save_submission(df_pred, filename=None):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        df_pred (pd.DataFrame): DataFrame containing ['id', 'cell_order'].
        filename (str, optional): Output filename. Defaults to Config.SUBMISSION_PATH.
    """
    if filename is None:
        filename = Config.SUBMISSION_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Ensure columns are correct
    if "id" not in df_pred.columns or "cell_order" not in df_pred.columns:
        raise ValueError("DataFrame must contain 'id' and 'cell_order' columns.")

    df_pred[["id", "cell_order"]].to_csv(filename, index=False)
