import json
import os
import pandas as pd
from bisect import bisect
from library.config import Config


def read_notebook(file_path):
    """
    Reads a notebook JSON file and returns code and markdown cells.

    The function relies on the fact that in the provided dataset, the 'cell_type' dictionary
    in the JSON preserves the correct relative order of code cells, while markdown cells
    are appended in a shuffled manner.

    Args:
        file_path (str): Path to the .json file.

    Returns:
        tuple: (code_cells, markdown_cells)
            - code_cells: List of tuples (cell_id, source_text).
            - markdown_cells: List of tuples (cell_id, source_text).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    code_cells = []
    markdown_cells = []

    # Iterate over keys; Python 3.7+ guarantees insertion order preservation.
    # Dataset spec implies code cells appear first in correct order.
    for cell_id, c_type in cell_types.items():
        source = sources.get(cell_id, "")
        # Ensure source is a string (handle potential list of strings format if it were to occur,
        # though dataset samples show strings)
        if isinstance(source, list):
            source = "".join(source)

        if c_type == "code":
            code_cells.append((cell_id, source))
        elif c_type == "markdown":
            markdown_cells.append((cell_id, source))

    return code_cells, markdown_cells


def preprocess_text(text):
    """
    Basic text preprocessing: lowercase and strip whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)
    return text.lower().strip()


def count_inversions(a):
    """
    Counts the number of inversions in a list using bisect.

    Args:
        a (list): List of comparable elements (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # bisect finds the index where x could be inserted while maintaining order.
        # Elements to the right of this index in 'sorted_so_far' are greater than x.
        # Since they were seen before x, they form inversions.
        idx = bisect(sorted_so_far, x)
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def kendall_tau(ground_truths, predictions):
    """
    Computes the Kendall Tau correlation metric for the competition.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        ground_truths (list of list of str): Correct cell orders (or space-delimited strings).
        predictions (list of list of str): Predicted cell orders (or space-delimited strings).

    Returns:
        float: The Kendall Tau score.
    """
    total_swaps = 0
    total_max_swaps = 0

    for gt, pred in zip(ground_truths, predictions):
        # Handle space-delimited strings if passed
        if isinstance(gt, str):
            gt = gt.split()
        if isinstance(pred, str):
            pred = pred.split()

        n = len(gt)
        if n <= 1:
            continue

        # Map ground truth IDs to their rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(gt)}

        # Convert prediction sequence to ranks based on ground truth
        # Filter out any IDs not in GT (though strictly there shouldn't be any)
        pred_ranks = [rank_map[cell_id] for cell_id in pred if cell_id in rank_map]

        # Calculate swaps (inversions) needed to sort pred_ranks to 0..n-1
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_max_swaps += n * (n - 1)

    if total_max_swaps == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_max_swaps)
    return score


def format_submission(ids, cell_orders, output_path=None):
    """
    Formats the predictions into a CSV file.

    Args:
        ids (list of str): Notebook IDs.
        cell_orders (list of str or list of list of str): Predicted orders.
        output_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure cell_orders are space-delimited strings
    formatted_orders = []
    for order in cell_orders:
        if isinstance(order, list):
            formatted_orders.append(" ".join(order))
        else:
            formatted_orders.append(order)

    df = pd.DataFrame({"id": ids, "cell_order": formatted_orders})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
