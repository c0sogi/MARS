import os
import json
import joblib
import numpy as np
import pandas as pd
from bisect import bisect
from library.config import Config


def count_inversions(a):
    """
    Counts the number of inversions in a list using a bisect-based approach.
    Complexity: O(N log N)
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to keep the list sorted
        idx = bisect(sorted_so_far, x)
        # The number of elements to the right of this position are greater than x
        # and thus form inversions with x.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau_metric(df_gt, preds_dict):
    """
    Computes the Kendall Tau correlation metric accumulated across the collection of notebooks.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_gt (pd.DataFrame): DataFrame containing ground truth. Must have columns:
                              - 'id': Notebook ID
                              - 'cell_order': Space-delimited string of correct cell order
        preds_dict (dict): Dictionary mapping notebook 'id' to a list of predicted cell IDs.

    Returns:
        float: The accumulated Kendall Tau score.
    """
    total_inversions = 0
    total_max_inversions = 0

    # Create a lookup for ground truth orders
    gt_dict = dict(zip(df_gt["id"], df_gt["cell_order"]))

    for nb_id, pred_order in preds_dict.items():
        if nb_id not in gt_dict:
            continue

        gt_order_str = gt_dict[nb_id]
        gt_order = gt_order_str.split()
        n = len(gt_order)

        # If a notebook has 0 or 1 cell, it doesn't contribute to the denominator (n*(n-1) = 0)
        # and requires 0 swaps.
        if n <= 1:
            continue

        # Map each cell_id to its correct rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Transform the predicted cell order into a list of ranks based on ground truth
        # We assume the prediction contains the same set of cells as the ground truth.
        # If prediction has missing/extra cells, this logic filters to intersection,
        # which is standard for this metric's evaluation context.
        pred_ranks = []
        for cell_id in pred_order:
            if cell_id in rank_map:
                pred_ranks.append(rank_map[cell_id])

        # The number of swaps needed to sort the predicted ranks is equal to the number of inversions
        s = count_inversions(pred_ranks)

        total_inversions += s
        total_max_inversions += n * (n - 1)

    # Avoid division by zero if all notebooks are trivial
    if total_max_inversions == 0:
        return 1.0

    k = 1 - 4 * (total_inversions / total_max_inversions)
    return k


def read_notebook_json(filepath):
    """
    Reads and parses a notebook JSON file.

    Args:
        filepath (str): The path to the .json file.

    Returns:
        dict: The parsed dictionary content of the notebook.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_artifacts(obj, path):
    """
    Saves a Python object (model, vectorizer, etc.) to disk.

    Handles directory creation and specific saving logic for LightGBM text models
    if the path ends in .txt, otherwise uses joblib.

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Check for LightGBM text model saving requirement
    if path.endswith(".txt"):
        # Case 1: Object is a LightGBM Booster
        if hasattr(obj, "save_model"):
            obj.save_model(path)
            return
        # Case 2: Object is a sklearn LGBM wrapper (LGBMRegressor/LGBMClassifier)
        # The underlying booster is stored in the 'booster_' attribute after fitting
        elif hasattr(obj, "booster_"):
            obj.booster_.save_model(path)
            return

    # Default: Use joblib for serialization
    joblib.dump(obj, path)
