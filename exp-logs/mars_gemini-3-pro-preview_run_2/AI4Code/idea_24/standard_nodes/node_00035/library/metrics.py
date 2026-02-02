import os
import pandas as pd
from bisect import bisect
from typing import List, Dict, Union
from library.config import SUBMISSION_DIR


def count_inversions(a: List[int]) -> int:
    """
    Counts the number of inversions in a list of integers using bisect.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].

    This function calculates the number of swaps required to sort the array,
    which is the numerator S in the Kendall Tau formula.

    Args:
        a: List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect returns the insertion point i such that all elements
        # to the left are <= x and all elements to the right are > x.
        # Since sorted_so_far contains elements that appeared *before* x in the original list,
        # elements to the right of the insertion point are those that are greater than x
        # but appeared before x, constituting an inversion.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(
    ground_truths: List[List[str]], predictions: List[List[str]]
) -> float:
    """
    Computes the global Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        ground_truths: List of lists, where each inner list contains the correct cell_id order.
        predictions: List of lists, where each inner list contains the predicted cell_id order.

    Returns:
        float: The Kendall Tau score.
    """
    total_inversions = 0
    total_pairs = 0  # This corresponds to sum of n*(n-1)

    for gt, pred in zip(ground_truths, predictions):
        n = len(gt)
        # If a notebook has 0 or 1 cell, it contributes 0 to the denominator (max swaps = 0).
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        rank_map = {cid: i for i, cid in enumerate(gt)}

        # Convert the predicted cell_id sequence to a rank sequence.
        # We assume predictions contain the same IDs as ground_truth.
        # Filter out any IDs not in ground truth to ensure robustness.
        pred_ranks = []
        for cid in pred:
            if cid in rank_map:
                pred_ranks.append(rank_map[cid])

        # Calculate swaps (inversions) needed to sort pred_ranks
        s = count_inversions(pred_ranks)

        total_inversions += s
        total_pairs += n * (n - 1)

    # Avoid division by zero if all notebooks are length 0 or 1
    if total_pairs == 0:
        return 1.0

    score = 1 - 4 * (total_inversions / total_pairs)
    return score


def compute_score(
    df_ground_truth: pd.DataFrame, predictions: Dict[str, List[str]]
) -> float:
    """
    Wrapper to compute score from a DataFrame and a prediction dictionary.

    Args:
        df_ground_truth: DataFrame containing 'id' and 'cell_order' columns.
        predictions: Dictionary mapping notebook 'id' to a list of cell_ids.

    Returns:
        float: The Kendall Tau score.
    """
    # Extract aligned lists
    gt_list = []
    pred_list = []

    # Iterate over the dataframe to ensure alignment and handle missing predictions
    for _, row in df_ground_truth.iterrows():
        nb_id = row["id"]
        gt_order_str = row["cell_order"]

        if nb_id not in predictions:
            continue

        gt_order = gt_order_str.split()
        pred_order = predictions[nb_id]

        # Ensure pred_order is a list of strings
        if isinstance(pred_order, str):
            pred_order = pred_order.split()

        gt_list.append(gt_order)
        pred_list.append(pred_order)

    score = compute_kendall_tau(gt_list, pred_list)

    # Print full precision as requested
    print(f"{score}")

    return score


def save_submission(
    predictions: Dict[str, List[str]], filename: str = "submission.csv"
) -> None:
    """
    Generates a submission CSV file in the correct format.

    Args:
        predictions: Dictionary mapping notebook 'id' to a list of cell_ids.
        filename: Name of the output file.
    """
    ids = []
    cell_orders = []

    for nb_id, cell_order in predictions.items():
        ids.append(nb_id)
        # Join list into space-delimited string if it's a list
        if isinstance(cell_order, list):
            cell_orders.append(" ".join(cell_order))
        else:
            cell_orders.append(cell_order)

    df_submission = pd.DataFrame({"id": ids, "cell_order": cell_orders})

    output_path = os.path.join(SUBMISSION_DIR, filename)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
