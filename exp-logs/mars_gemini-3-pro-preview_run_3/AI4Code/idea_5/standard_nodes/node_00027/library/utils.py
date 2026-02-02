import os
import random
import numpy as np
import torch
from bisect import bisect
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CuDNN for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_inversions(a):
    """
    Counts the number of inversions in a list using the bisect module.
    An inversion is a pair of elements (a[i], a[j]) such that i < j and a[i] > a[j].
    This represents the minimum number of swaps of adjacent entries needed to sort the list.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to keep the list sorted
        idx = bisect(sorted_so_far, x)
        # The number of elements greater than x that have already been processed
        # is equal to the number of elements currently in the list minus the insertion index.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(df_pred, df_gt):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    Formula: K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))

    Where:
        S_i is the number of swaps needed to sort the predicted order for notebook i.
        n_i is the number of cells in notebook i.

    Args:
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns for predictions.
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns for ground truth.

    Returns:
        float: The computed Kendall Tau score.
    """
    # Merge predictions with ground truth on notebook ID
    df = df_pred.merge(df_gt, on="id", suffixes=("_pred", "_gt"))

    total_swaps = 0
    total_possible = 0

    for _, row in df.iterrows():
        pred_order_str = row["cell_order_pred"]
        gt_order_str = row["cell_order_gt"]

        # Split strings into lists of cell IDs
        pred_order = pred_order_str.split()
        gt_order = gt_order_str.split()

        # Create a mapping from cell ID to its rank (index) in the ground truth
        gt_ranks = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert the predicted cell order into a list of ranks based on ground truth
        # Filter out any predicted cells that are not in the ground truth
        pred_ranks = [
            gt_ranks[cell_id] for cell_id in pred_order if cell_id in gt_ranks
        ]

        n = len(pred_ranks)

        # If there are fewer than 2 cells, no order to predict (or trivial)
        if n < 2:
            continue

        # Calculate the number of swaps (inversions) needed to sort the predicted ranks
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    # Avoid division by zero
    if total_possible == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score
