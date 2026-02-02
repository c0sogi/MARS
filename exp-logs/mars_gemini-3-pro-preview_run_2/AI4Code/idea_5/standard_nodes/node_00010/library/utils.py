import os
import random
import numpy as np
import torch
from bisect import bisect
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_inversions(a):
    """
    Counts the number of inversions in a list using a bisect-based approach.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    Complexity: O(N log N)
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to keep the list sorted
        idx = bisect(sorted_so_far, x)
        # All elements currently in sorted_so_far at indices >= idx are greater than x
        # and appeared before x, so they form inversions.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau_metric(df, preds):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'cell_order' columns.
                           'cell_order' should be a space-delimited string of cell IDs.
        preds (dict): Dictionary mapping notebook 'id' to a list of predicted cell_ids (strings).

    Returns:
        float: The global Kendall Tau score accumulated across the dataset.
    """
    total_inversions = 0
    total_n_comb = 0

    for _, row in df.iterrows():
        nid = row["id"]

        # Parse ground truth order
        if isinstance(row["cell_order"], str):
            gt_order = row["cell_order"].split()
        else:
            gt_order = list(row["cell_order"])

        n = len(gt_order)

        # Notebooks with fewer than 2 cells have no pairs to order
        if n < 2:
            continue

        # Retrieve prediction
        if nid not in preds:
            # If a notebook is missing from predictions, we skip it to avoid errors.
            # In a real submission, this would be a critical issue.
            continue

        pred_order = preds[nid]

        # Map ground truth cell IDs to their rank (0 to n-1)
        gt_ranks = {cid: i for i, cid in enumerate(gt_order)}

        # Construct the permutation vector based on the predicted order.
        # We iterate through the PREDICTED order and append the TRUE rank of each cell.
        # If the prediction perfectly matches the ground truth, this list will be [0, 1, 2, ...].
        permutation = []
        for cid in pred_order:
            if cid in gt_ranks:
                permutation.append(gt_ranks[cid])

        # Count swaps (inversions) needed to sort this permutation
        s = count_inversions(permutation)

        total_inversions += s
        total_n_comb += n * (n - 1)

    if total_n_comb == 0:
        return 0.0

    # Competition Metric Formula: K = 1 - 4 * (Sum S_i) / (Sum n_i * (n_i - 1))
    return 1.0 - 4.0 * total_inversions / total_n_comb
