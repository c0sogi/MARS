import os
import random
import bisect
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_inversions(prediction, ground_truth):
    """
    Counts the number of inversions required to transform the prediction order
    into the ground truth order. This is equivalent to the number of swaps
    of adjacent entries needed to sort the predicted order.

    Args:
        prediction (list): List of cell IDs in predicted order.
        ground_truth (list): List of cell IDs in correct order.

    Returns:
        tuple: (number of inversions, length of sequence)
    """
    # Map ground truth cell IDs to their rank (0, 1, 2, ...)
    gt_rank = {cell_id: i for i, cell_id in enumerate(ground_truth)}

    # Convert prediction sequence to a list of ranks based on ground truth
    # We only consider cells that exist in the ground truth
    pred_ranks = [gt_rank[cell_id] for cell_id in prediction if cell_id in gt_rank]

    n = len(pred_ranks)
    if n < 2:
        return 0, n

    # Use bisect for efficient inversion counting.
    # We iterate through the sequence and maintain a sorted list of elements seen so far.
    # For each element, the number of elements already seen that are *greater* than it
    # contributes to the inversion count.
    inversions = 0
    seen = []

    for rank in pred_ranks:
        # bisect_right returns the insertion point after any existing entries of 'rank'.
        # Elements currently in 'seen' at indices >= idx are greater than 'rank'.
        idx = bisect.bisect_right(seen, rank)
        inversions += len(seen) - idx
        bisect.insort(seen, rank)

    return inversions, n


def compute_kendall_tau(df_pred, df_gt):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))

    Where S_i is the number of swaps (inversions) for notebook i,
    and n_i is the number of cells in notebook i.

    Args:
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).
        df_gt (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).

    Returns:
        float: The computed Kendall Tau score.
    """
    # Merge on ID to ensure alignment of predictions and ground truth
    merged = df_pred.merge(df_gt, on="id", suffixes=("_pred", "_gt"))

    total_inversions = 0
    total_pairs = 0  # This accumulates sum(n * (n - 1))

    for _, row in merged.iterrows():
        pred_list = row["cell_order_pred"].split()
        gt_list = row["cell_order_gt"].split()

        inversions, n = count_inversions(pred_list, gt_list)

        total_inversions += inversions
        total_pairs += n * (n - 1)

    if total_pairs == 0:
        return 1.0

    score = 1 - 4 * (total_inversions / total_pairs)
    return score
