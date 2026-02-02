import os
import json
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_notebook(file_path_rel):
    """
    Reads a notebook JSON file from the input directory.

    Args:
        file_path_rel (str): Relative path to the notebook file (e.g., 'train/00001756c60be8.json').

    Returns:
        dict: The content of the notebook as a dictionary.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path_rel)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_text(text):
    """
    Cleans and preprocesses source text from a notebook cell.

    Args:
        text (str): The raw text content of a cell.

    Returns:
        str: The cleaned and truncated text.
    """
    if text is None:
        return ""

    # Convert to string and strip whitespace
    text = str(text).strip()

    # Lowercase for semantic consistency
    text = text.lower()

    # Truncate to a reasonable character limit to handle outliers
    # (Analysis showed some cells > 200k chars).
    # 2048 chars is sufficient for the semantic backbone (limit 128 tokens).
    return text[:2048]


def _count_inversions(arr):
    """
    Counts the number of inversions in an array.
    An inversion is a pair (i, j) such that i < j and arr[i] > arr[j].
    This represents the number of swaps needed to sort the array.
    """
    n = len(arr)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count


def compute_kendall_tau(df_ground_truth, predictions):
    """
    Computes the Kendall Tau correlation metric accumulated across the collection.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n(n-1))

    Args:
        df_ground_truth (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns.
        predictions (dict or pd.DataFrame):
            If dict: {notebook_id: [list_of_cell_ids] or "space separated string"}
            If DataFrame: Must have 'id' and 'cell_order' columns.

    Returns:
        float: The accumulated Kendall Tau score.
    """
    # Convert predictions DataFrame to dict if necessary
    if isinstance(predictions, pd.DataFrame):
        predictions = dict(zip(predictions["id"], predictions["cell_order"]))

    total_inversions = 0
    total_n_pairs = 0  # Accumulator for sum(n * (n - 1))

    for _, row in df_ground_truth.iterrows():
        nb_id = row["id"]

        # Skip if no prediction for this notebook
        if nb_id not in predictions:
            continue

        # Parse ground truth
        true_order = row["cell_order"].split()
        n = len(true_order)

        # If notebook has 0 or 1 cell, it contributes nothing to the score denominator
        if n <= 1:
            continue

        # Parse prediction
        pred_order = predictions[nb_id]
        if isinstance(pred_order, str):
            pred_order = pred_order.split()

        # Map cell IDs to their rank (index) in the ground truth
        # We only care about the relative order of cells that exist in the ground truth.
        cell_rank_map = {cell_id: rank for rank, cell_id in enumerate(true_order)}

        # Convert the predicted cell order to a list of true ranks
        # Filter out any predicted cells that aren't in the ground truth (though valid submission shouldn't have this)
        predicted_ranks = [
            cell_rank_map[cid] for cid in pred_order if cid in cell_rank_map
        ]

        # Calculate number of swaps (inversions) needed to sort the predicted ranks
        # to match the ground truth (which would be 0, 1, 2, ...)
        swaps = _count_inversions(predicted_ranks)

        total_inversions += swaps
        total_n_pairs += n * (n - 1)

    if total_n_pairs == 0:
        return 0.0

    # Calculate final metric
    kendall_tau = 1 - 4 * (total_inversions / total_n_pairs)
    return kendall_tau
