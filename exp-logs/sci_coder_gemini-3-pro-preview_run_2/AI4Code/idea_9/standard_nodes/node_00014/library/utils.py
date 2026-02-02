import os
import random
import numpy as np
import torch
import pandas as pd
from bisect import bisect
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _count_inversions(a):
    """
    Efficient inversion counting using bisect (insertion sort simulation).
    Calculates the number of swaps needed to sort the array.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to keep sorted_so_far sorted
        idx = bisect(sorted_so_far, x)
        # The number of elements greater than x (which are already in sorted_so_far)
        # contributes to the inversion count.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(predictions_df, ground_truth_df):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.
    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        predictions_df (pd.DataFrame): DataFrame containing 'id' and 'cell_order'.
        ground_truth_df (pd.DataFrame): DataFrame containing 'id' and 'cell_order'.

    Returns:
        float: The Kendall Tau score.
    """
    # Ensure IDs are strings for accurate merging
    predictions_df = predictions_df.copy()
    ground_truth_df = ground_truth_df.copy()
    predictions_df["id"] = predictions_df["id"].astype(str)
    ground_truth_df["id"] = ground_truth_df["id"].astype(str)

    # Merge predictions with ground truth on notebook ID
    merged = pd.merge(
        predictions_df, ground_truth_df, on="id", suffixes=("_pred", "_gt")
    )

    total_swaps = 0
    total_possible = 0

    for _, row in merged.iterrows():
        pred_str = row["cell_order_pred"]
        gt_str = row["cell_order_gt"]

        if pd.isna(pred_str) or pd.isna(gt_str):
            continue

        pred_order = pred_str.split()
        gt_order = gt_str.split()

        n = len(gt_order)
        if n < 2:
            continue

        # Map ground truth cell IDs to ranks 0 to n-1
        gt_rank_map = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert prediction to ranks based on GT
        # Only consider cells that exist in the ground truth
        pred_ranks = []
        for cell_id in pred_order:
            if cell_id in gt_rank_map:
                pred_ranks.append(gt_rank_map[cell_id])

        # Calculate swaps (inversions) needed to sort pred_ranks to 0..n-1
        swaps = _count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible += n * (n - 1)

    if total_possible == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible)
    return score


def validate_paths():
    """
    Validates that necessary directories and files exist based on Config.
    Raises FileNotFoundError if any required path is missing.
    """
    paths_to_check = [
        Config.INPUT_DIR,
        Config.METADATA_DIR,
        Config.TRAIN_METADATA_PATH,
        Config.VAL_METADATA_PATH,
        Config.TEST_METADATA_PATH,
    ]

    for path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")
