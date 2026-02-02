import os
import pandas as pd
from bisect import bisect
from library.config import Config


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers.
    An inversion is a pair (i, j) such that i < j but a[i] > a[j].
    This represents the number of swaps needed to sort the array.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # The number of elements seen so far that are greater than x
        # is the length of the sorted list minus the insertion index of x.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau_metric(df_pred, df_gt):
    """
    Computes the Kendall Tau correlation metric as defined in the competition.

    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_pred (pd.DataFrame): DataFrame with 'id' and 'cell_order' columns (predictions).
        df_gt (pd.DataFrame): DataFrame with 'id' and 'cell_order' columns (ground truth).

    Returns:
        float: The Kendall tau correlation score.
    """
    # Ensure inputs are copies to avoid modifying originals
    preds = df_pred.set_index("id")["cell_order"].to_dict()
    gts = df_gt.set_index("id")["cell_order"].to_dict()

    total_swaps = 0
    total_possible_pairs = 0

    # Iterate over the intersection of IDs
    common_ids = set(preds.keys()).intersection(set(gts.keys()))

    for key in common_ids:
        pred_order_str = preds[key]
        gt_order_str = gts[key]

        # Convert space-delimited strings to lists
        pred_order = pred_order_str.split()
        gt_order = gt_order_str.split()

        n = len(gt_order)
        if n < 2:
            continue

        # Create a rank mapping from the ground truth
        # cell_id -> correct rank (0 to n-1)
        gt_ranks = {cell_id: i for i, cell_id in enumerate(gt_order)}

        # Convert prediction to a list of ranks based on GT
        # Filter out any predicted cells that aren't in GT (though strictly shouldn't happen)
        pred_ranks = [
            gt_ranks[cell_id] for cell_id in pred_order if cell_id in gt_ranks
        ]

        # Calculate swaps (inversions) needed to sort pred_ranks into 0..n-1
        swaps = count_inversions(pred_ranks)

        # Calculate worst-case swaps for this notebook size
        # n * (n - 1) is the denominator term in the formula
        # (Note: max swaps is n(n-1)/2, but the formula uses n(n-1) in denominator with factor 4)
        max_pairs = n * (n - 1)

        total_swaps += swaps
        total_possible_pairs += max_pairs

    if total_possible_pairs == 0:
        return 0.0

    # Formula: K = 1 - 4 * (Sum S_i) / (Sum n_i * (n_i - 1))
    score = 1 - 4 * (total_swaps / total_possible_pairs)
    return score


def save_submission(ids, cell_orders):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list): List of notebook IDs.
        cell_orders (list): List of lists, where each inner list contains cell IDs in predicted order.
    """
    # Convert lists of cell IDs to space-delimited strings
    cell_order_strings = [" ".join(order) for order in cell_orders]

    df_submission = pd.DataFrame({"id": ids, "cell_order": cell_order_strings})

    output_path = Config.SUBMISSION_PATH
    # Ensure directory exists (handled by Config usually, but good practice)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def load_data(path):
    """
    Simple wrapper to load CSV data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
