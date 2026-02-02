import os
import json
import re
import pandas as pd
import numpy as np
from bisect import bisect_left
from typing import List, Dict, Any, Callable, Optional, Union

from library.config import Config


def preprocess_text(text: str) -> str:
    """
    Performs basic text preprocessing.
    Converts text to lowercase. Does not strip accents, preserving them for TF-IDF.
    """
    if not text:
        return ""
    return text.lower()


def extract_symbolic_tokens(text: str) -> List[str]:
    """
    Extracts symbolic identifiers (variables, function names) from text
    using the regex pattern defined in the configuration.
    """
    if not text:
        return []
    pattern = Config.SYMBOLIC_TOKEN_PATTERN
    return re.findall(pattern, text)


def get_ranks(cell_order: List[str]) -> Dict[str, float]:
    """
    Converts an ordered list of cell IDs into a dictionary of normalized ranks.

    The rank is calculated as position / (n - 1), resulting in a value between 0.0 and 1.0.
    If the notebook has 0 or 1 cells, the rank is 0.0.

    Args:
        cell_order: List of cell IDs representing the correct order.

    Returns:
        Dictionary mapping cell_id to normalized rank.
    """
    n = len(cell_order)
    if n <= 1:
        return {cid: 0.0 for cid in cell_order}

    ranks = {}
    for i, cid in enumerate(cell_order):
        ranks[cid] = i / (n - 1)
    return ranks


def _count_inversions(arr: List[int]) -> int:
    """
    Counts the number of inversions in a list of integers using a bisect-based approach.
    An inversion is a pair (i, j) such that i < j and arr[i] > arr[j].

    Complexity: O(N log N)
    """
    inversions = 0
    sorted_seen = []
    for x in arr:
        # Find the position where x should be inserted to maintain order
        idx = bisect_left(sorted_seen, x)
        # Elements currently in sorted_seen at indices >= idx are greater than x
        # These elements appeared before x in the original array, so they form inversions.
        inversions += len(sorted_seen) - idx
        sorted_seen.insert(idx, x)
    return inversions


def kendall_tau(ground_truths: List[List[str]], predictions: List[List[str]]) -> float:
    """
    Computes the Kendall Tau correlation metric accumulated across a collection of notebooks.

    Formula: K = 1 - 4 * (Sum(S_i) / Sum(n_i * (n_i - 1)))
    Where S_i is the number of swaps needed to sort the prediction into the ground truth.

    Args:
        ground_truths: List of lists, where each inner list is the correct cell order.
        predictions: List of lists, where each inner list is the predicted cell order.

    Returns:
        The global Kendall Tau score.
    """
    total_swaps = 0
    total_pairs = 0

    for gt, pred in zip(ground_truths, predictions):
        n = len(gt)
        if n <= 1:
            continue

        # Map ground truth cell IDs to their correct rank index (0 to n-1)
        gt_rank_map = {cid: i for i, cid in enumerate(gt)}

        # Transform the prediction list into a list of ranks based on ground truth
        # Filter out any IDs in prediction that are not in ground truth (safety)
        pred_ranks = []
        for cid in pred:
            if cid in gt_rank_map:
                pred_ranks.append(gt_rank_map[cid])

        # Calculate swaps (inversions) needed to sort the predicted ranks
        swaps = _count_inversions(pred_ranks)

        total_swaps += swaps
        total_pairs += n * (n - 1)

    if total_pairs == 0:
        return 1.0

    return 1 - 4 * (total_swaps / total_pairs)


def read_notebook(filepath: str) -> Dict[str, Any]:
    """
    Reads a JSON notebook file from the given path.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_process_data(
    cache_path: str,
    process_fn: Callable[..., pd.DataFrame],
    load_cached_data: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Handles caching for deterministic data processing steps.

    Logic:
    1. If load_cached_data is True and cache_path exists, load and return data (Parquet).
    2. Otherwise, execute process_fn(**kwargs).
    3. Save the result to cache_path (creating directories if needed).
    4. Return the result.

    Args:
        cache_path: Path to the parquet file for caching.
        process_fn: Function to generate the data if cache is missed.
        load_cached_data: Boolean flag to enable/disable loading from cache.
        **kwargs: Arguments passed to process_fn.

    Returns:
        The processed DataFrame.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception:
            # If load fails, fall back to processing
            pass

    # print(f"Computing data using {process_fn.__name__}...")
    df = process_fn(**kwargs)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved data to cache at {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df
