import os
import json
import random
import numpy as np
import pandas as pd
import joblib
import torch
from bisect import bisect
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_joblib(obj, path):
    """
    Saves a Python object using joblib.
    Ensures the parent directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    """
    Loads a Python object using joblib.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)


def count_inversions(a):
    """
    Counts the number of inversions in a list.
    An inversion is a pair of elements (a[i], a[j]) such that i < j and a[i] > a[j].
    This is equivalent to the number of swaps needed to sort the array.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect returns the insertion point to maintain sorted order.
        # Elements currently in sorted_so_far to the right of this index are larger than x,
        # and thus form inversions with x.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(df_gt, df_pred):
    """
    Computes the global Kendall Tau correlation metric as defined in the task.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_gt: DataFrame containing ['id', 'cell_order'] for ground truth.
        df_pred: DataFrame containing ['id', 'cell_order'] for predictions.

    Returns:
        float: The weighted Kendall Tau score.
    """
    gt_dict = dict(zip(df_gt["id"], df_gt["cell_order"]))
    pred_dict = dict(zip(df_pred["id"], df_pred["cell_order"]))

    total_swaps = 0
    total_possible = 0

    # Iterate over notebooks present in both (intersection)
    common_ids = set(gt_dict.keys()).intersection(set(pred_dict.keys()))

    for nb_id in common_ids:
        gt_order = gt_dict[nb_id].split()
        pred_order = pred_dict[nb_id].split()

        n = len(gt_order)
        if n <= 1:
            continue

        # Map cell_id to its rank in the ground truth
        gt_ranks = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Transform predictions into a list of ranks based on ground truth
        # We filter to ensure we only consider cells that exist in the ground truth
        pred_ranks = [
            gt_ranks[cell_id] for cell_id in pred_order if cell_id in gt_ranks
        ]

        # The number of swaps to sort pred_ranks is the number of inversions
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    if total_possible == 0:
        return 1.0

    return 1 - 4 * (total_swaps / total_possible)


def read_notebook(filepath):
    """
    Reads a JSON notebook file.
    Resolves the path relative to INPUT_DIR if it doesn't exist as absolute.
    """
    if not os.path.exists(filepath):
        filepath = os.path.join(Config.INPUT_DIR, filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data_splits():
    """
    Loads the train, validation, and test metadata DataFrames.
    """
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    return df_train, df_val, df_test
