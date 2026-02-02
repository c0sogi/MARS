import re
import bisect
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

# Initialize stemmer globally to avoid initialization overhead on repeated calls
_stemmer = PorterStemmer()


def preprocess_text(text, stem=True):
    """
    Cleans and preprocesses text for TF-IDF vectorization or other NLP tasks.

    Pipeline:
    1. Lowercase conversion.
    2. Removal of non-alphanumeric characters (preserving spaces).
    3. (Optional) Stemming using PorterStemmer.

    Args:
        text (str): Input source code or markdown text.
        stem (bool): Whether to apply stemming. Defaults to True.

    Returns:
        str: Processed text string. Returns empty string if input is invalid.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters (keep spaces)
    # Replaces anything that is not a letter, digit, or whitespace with a space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces into one and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if stem:
        # Tokenize by splitting on whitespace and apply stemming
        tokens = text.split()
        stemmed_tokens = [_stemmer.stem(t) for t in tokens]
        return " ".join(stemmed_tokens)

    return text


def count_inversions(arr):
    """
    Counts the number of inversions in a list of ranks.
    An inversion is a pair of indices (i, j) such that i < j and arr[i] > arr[j].
    This is equivalent to the number of swaps needed to sort the array using bubble sort.

    Args:
        arr (list[int]): List of integers representing ranks.

    Returns:
        int: Total number of inversions.
    """
    inversions = 0
    sorted_arr = []

    # Iterate through the array and build a sorted version incrementally
    for x in arr:
        # bisect_right returns the insertion point to maintain sorted order.
        # Elements already in sorted_arr at indices >= idx are greater than x.
        # Since they appeared earlier in the input 'arr', they form inversions with x.
        idx = bisect.bisect_right(sorted_arr, x)

        # Add the number of elements greater than x to the inversion count
        inversions += len(sorted_arr) - idx

        # Insert x into the sorted list
        sorted_arr.insert(idx, x)

    return inversions


def compute_kendall_tau(df_pred, df_true):
    """
    Computes the Kendall Tau correlation metric accumulated across a collection of notebooks.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n * (n - 1))

    Args:
        df_pred (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).
        df_true (pd.DataFrame): DataFrame containing 'id' and 'cell_order' (space-delimited string).

    Returns:
        float: The calculated Kendall Tau score.
    """
    # Work on copies to prevent side effects
    df_pred = df_pred.copy()
    df_true = df_true.copy()

    # Merge predictions with ground truth on 'id'
    # Use inner join to evaluate only on notebooks present in both sets
    merged = pd.merge(df_true, df_pred, on="id", suffixes=("_true", "_pred"))

    total_swaps = 0
    total_denom = 0

    for _, row in merged.iterrows():
        true_order_str = row["cell_order_true"]
        pred_order_str = row["cell_order_pred"]

        if not isinstance(true_order_str, str) or not isinstance(pred_order_str, str):
            continue

        true_order = true_order_str.split()
        pred_order = pred_order_str.split()

        n = len(true_order)

        # If n < 2, the denominator n(n-1) is 0. These cases do not contribute to the metric.
        if n < 2:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(true_order)}

        # Convert the predicted cell order into a list of ranks based on ground truth
        # Filter to ensure we only consider cells that exist in the ground truth
        pred_ranks = [
            rank_map[cell_id] for cell_id in pred_order if cell_id in rank_map
        ]

        # Calculate the number of swaps (inversions) for this notebook
        swaps = count_inversions(pred_ranks)

        # Accumulate totals
        total_swaps += swaps
        total_denom += n * (n - 1)

    # Avoid division by zero if the dataset is empty or all notebooks are trivial
    if total_denom == 0:
        return 1.0

    kendall_tau = 1 - 4 * (total_swaps / total_denom)
    return kendall_tau
