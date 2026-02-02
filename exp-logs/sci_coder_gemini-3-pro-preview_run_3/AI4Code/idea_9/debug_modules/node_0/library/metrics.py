import pandas as pd
import numpy as np
import bisect
from typing import List, Union


def count_inversions(prediction: List[str], ground_truth: List[str]) -> int:
    """
    Counts the number of swaps (inversions) needed to sort the prediction
    into the ground truth order.

    This is equivalent to counting the number of inversions in the sequence
    of ranks when the prediction IDs are mapped to their positions in the
    ground truth.

    Args:
        prediction: List of cell IDs in the predicted order.
        ground_truth: List of cell IDs in the correct order.

    Returns:
        int: The number of inversions.
    """
    # Map cell IDs to their ground truth rank (0 to n-1)
    gt_rank = {uid: i for i, uid in enumerate(ground_truth)}

    # Convert prediction sequence to a list of ranks.
    # We filter to ensure we only consider cells present in the ground truth
    # to handle potential inconsistencies safely, though valid submissions
    # should match exactly.
    pred_ranks = [gt_rank[uid] for uid in prediction if uid in gt_rank]

    inversions = 0
    sorted_seen = []

    # Iterate through the predicted ranks.
    # For each element, we want to count how many elements *already processed*
    # (i.e., to the left) are greater than the current element.
    # This is equivalent to counting pairs (i, j) such that i < j and A[i] > A[j].
    for r in pred_ranks:
        # Find the position where 'r' would fit in the sorted list of seen elements.
        # bisect_right returns an insertion point after all existing entries of 'r'.
        idx = bisect.bisect_right(sorted_seen, r)

        # The number of elements in sorted_seen that are strictly greater than 'r'
        # is the total length minus the insertion index.
        # These elements appear before 'r' in the prediction but have a higher rank,
        # constituting inversions.
        inversions += len(sorted_seen) - idx

        # Insert 'r' into the sorted list to maintain the invariant for the next step.
        sorted_seen.insert(idx, r)

    return inversions


def kendall_tau_metric(df_preds: pd.DataFrame, df_truth: pd.DataFrame) -> float:
    """
    Computes the Kendall tau correlation metric as defined in the competition.

    The metric is defined as:
    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Where:
    - Sum of Swaps is the total number of inversions accumulated across all notebooks.
    - Sum of n*(n-1) is the worst-case number of swaps accumulated across all notebooks
      (multiplied by 2, as worst case is n(n-1)/2).

    Args:
        df_preds: DataFrame with columns ['id', 'cell_order'].
                  'cell_order' should be a space-delimited string of cell IDs.
        df_truth: DataFrame with columns ['id', 'cell_order'].

    Returns:
        float: The Kendall tau score accumulated across the dataset.
    """
    # Merge predictions and ground truth on notebook id to ensure alignment
    df = pd.merge(
        df_preds[["id", "cell_order"]],
        df_truth[["id", "cell_order"]],
        on="id",
        suffixes=("_pred", "_gt"),
    )

    total_swaps = 0
    total_possible_pairs = 0

    for _, row in df.iterrows():
        pred_order = row["cell_order_pred"].split()
        gt_order = row["cell_order_gt"].split()

        n = len(gt_order)

        # If a notebook has fewer than 2 cells, no pairs exist, so it contributes
        # nothing to the numerator or denominator.
        if n < 2:
            continue

        swaps = count_inversions(pred_order, gt_order)

        total_swaps += swaps
        # The formula denominator term for this notebook is n * (n - 1)
        total_possible_pairs += n * (n - 1)

    # Handle edge case where no valid pairs exist in the entire dataset
    if total_possible_pairs == 0:
        return 0.0

    score = 1 - 4 * (total_swaps / total_possible_pairs)
    return score
