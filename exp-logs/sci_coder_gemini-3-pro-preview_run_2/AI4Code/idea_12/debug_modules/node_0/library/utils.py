import os
import random
import numpy as np
import pandas as pd
import torch
from bisect import bisect, insort
from typing import Dict, List, Optional
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_inversions(a: List[int]) -> int:
    """
    Calculates the number of inversions in a list of integers using bisect.
    Used as a helper for the Kendall Tau metric.

    Args:
        a: List of integers (ranks).

    Returns:
        Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect returns the insertion point to maintain order.
        # Elements to the right of this point in 'sorted_so_far' are greater than x.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        insort(sorted_so_far, x)
    return inversions


def kendall_tau_metric(df_gt: pd.DataFrame, preds_dict: Dict[str, List[str]]) -> float:
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Inversions) / (Sum of n(n-1))

    Args:
        df_gt: DataFrame containing ground truth with columns ['id', 'cell_order'].
        preds_dict: Dictionary mapping notebook_id to a list of ordered cell_ids.

    Returns:
        The accumulated Kendall Tau score.
    """
    total_inversions = 0
    total_max_swaps = 0  # This is Sum of n(n-1)

    # Iterate over the ground truth DataFrame
    for _, row in df_gt.iterrows():
        nb_id = row["id"]

        # Skip if no prediction available for this ID (e.g., when evaluating on a subset)
        if nb_id not in preds_dict:
            continue

        gt_order = row["cell_order"].split()
        pred_order = preds_dict[nb_id]

        n = len(gt_order)
        # If a notebook has 0 or 1 cell, it contributes 0 to swaps and 0 to max_swaps.
        if n <= 1:
            continue

        # Map cell IDs to their correct rank (0 to n-1)
        gt_rank_map = {cid: i for i, cid in enumerate(gt_order)}

        # Convert the predicted cell order into a list of ranks.
        # We filter to ensure we only consider cells that exist in the ground truth.
        pred_ranks = []
        for cid in pred_order:
            if cid in gt_rank_map:
                pred_ranks.append(gt_rank_map[cid])

        # Calculate inversions (swaps needed to sort pred_ranks to 0..n-1)
        s = count_inversions(pred_ranks)

        total_inversions += s
        total_max_swaps += n * (n - 1)

    # Avoid division by zero
    if total_max_swaps == 0:
        return 1.0

    # The metric formula: K = 1 - 4 * (S / Sum(n(n-1)))
    k = 1.0 - 4.0 * (total_inversions / total_max_swaps)
    return k


def format_submission(
    preds_dict: Dict[str, List[str]], save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Formats the predictions into the required submission DataFrame format.

    Args:
        preds_dict: Dictionary mapping notebook_id to a list of ordered cell_ids.
        save_path: Optional path to save the CSV file.

    Returns:
        DataFrame with columns ['id', 'cell_order'].
    """
    ids = []
    orders = []

    for nb_id, order_list in preds_dict.items():
        ids.append(nb_id)
        # Join the list of cell IDs with spaces
        orders.append(" ".join(order_list))

    df_submission = pd.DataFrame({"id": ids, "cell_order": orders})

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_submission.to_csv(save_path, index=False)

    return df_submission
