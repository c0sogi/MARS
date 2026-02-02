import os
import json
import pandas as pd
from library.config import Config


def read_notebook(filepath):
    """
    Reads a JSON notebook file from the input directory.

    Args:
        filepath (str): Relative path to the notebook file (e.g., 'train/id.json').
                        Can be obtained from the metadata dataframes.

    Returns:
        dict: The parsed JSON content of the notebook.
    """
    full_path = os.path.join(Config.INPUT_DIR, filepath)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_inversions(arr):
    """
    Counts the number of inversions in a list of ranks.

    An inversion is a pair of elements (arr[i], arr[j]) such that i < j
    and arr[i] > arr[j].

    Args:
        arr (list[int]): List of ranks.

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    n = len(arr)
    # Using nested loops (O(N^2)) is sufficient as N (cells per notebook) is small.
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions


def kendall_tau_metric(df_gt, preds):
    """
    Computes the Kendall Tau correlation metric accumulated across the collection.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_gt (pd.DataFrame): Ground truth DataFrame containing columns ['id', 'cell_order'].
                              'cell_order' should be a space-delimited string of cell IDs.
        preds (dict or pd.Series): Predictions mapping notebook_id to predicted cell order.
                                   Values can be a list of cell IDs or a space-delimited string.

    Returns:
        float: The global Kendall Tau correlation score.
    """
    total_inversions = 0
    total_n_comb = 0

    # Create a dictionary for fast ground truth lookup: id -> list of cell_ids
    gt_dict = dict(zip(df_gt["id"], df_gt["cell_order"]))

    for nb_id, gt_order_str in gt_dict.items():
        # Retrieve prediction
        if isinstance(preds, pd.Series):
            if nb_id not in preds.index:
                continue
            pred_order = preds.loc[nb_id]
        elif isinstance(preds, dict):
            if nb_id not in preds:
                continue
            pred_order = preds[nb_id]
        else:
            raise ValueError("preds must be a dict or pd.Series")

        # Normalize ground truth to list
        if isinstance(gt_order_str, str):
            gt_cells = gt_order_str.split()
        else:
            gt_cells = list(gt_order_str)

        n = len(gt_cells)
        if n <= 1:
            # No pairs to swap, contributes nothing to denominator
            continue

        # Normalize prediction to list
        if isinstance(pred_order, str):
            pred_cells = pred_order.split()
        else:
            pred_cells = list(pred_order)

        # Map ground truth cell IDs to their correct rank (0, 1, 2, ...)
        cell_to_rank = {cell_id: i for i, cell_id in enumerate(gt_cells)}

        # Convert the predicted sequence of cell IDs into a sequence of ranks.
        # We filter for cells that exist in the ground truth to handle potential mismatches safely.
        pred_ranks = [
            cell_to_rank[cell_id] for cell_id in pred_cells if cell_id in cell_to_rank
        ]

        # Calculate number of swaps (inversions) needed to sort the predicted ranks
        inversions = _count_inversions(pred_ranks)

        total_inversions += inversions
        total_n_comb += n * (n - 1)

    if total_n_comb == 0:
        return 0.0

    score = 1 - 4 * (total_inversions / total_n_comb)
    return score


def format_submission(ids, cell_orders, output_path=Config.SUBMISSION_PATH):
    """
    Formats the predictions and saves them to a CSV file in the required format.

    Args:
        ids (list): List of notebook IDs.
        cell_orders (list): List of predicted cell orders. Each element can be
                            a list of cell IDs or a space-delimited string.
        output_path (str): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
    """
    formatted_orders = []
    for order in cell_orders:
        if isinstance(order, list):
            formatted_orders.append(" ".join(order))
        else:
            formatted_orders.append(str(order))

    df_submission = pd.DataFrame({"id": ids, "cell_order": formatted_orders})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
