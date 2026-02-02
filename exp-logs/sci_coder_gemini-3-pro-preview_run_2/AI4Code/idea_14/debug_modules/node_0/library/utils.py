import os
import json
import re
import pandas as pd
import numpy as np
from library.config import Config


def preprocess_text(text):
    """
    Performs basic text cleaning: converts to lowercase and strips whitespace.
    Useful for standardizing text before vectorization.
    """
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def extract_identifiers(text):
    """
    Extracts variable names and identifiers from text using regex.
    Returns a set of unique identifiers found in the text.
    This supports the 'Symbolic Anchoring' view by capturing code references.
    """
    if not isinstance(text, str):
        return set()
    # Regex pattern to capture standard variable names (e.g., my_var, Model_1)
    pattern = r"[a-zA-Z_][a-zA-Z0-9_]*"
    tokens = re.findall(pattern, text)
    return set(tokens)


def read_notebook(filepath):
    """
    Reads a JSON notebook file from the input directory.

    Args:
        filepath (str): Relative path to the notebook (e.g., 'train/id.json').

    Returns:
        dict: The notebook content as a dictionary.
    """
    full_path = os.path.join(Config.INPUT_DIR, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # Return empty structure on failure to prevent pipeline crash
        return {}


def count_inversions(arr):
    """
    Counts the number of inversions in a list of ranks.
    Used as a helper for the Kendall Tau calculation.
    Time Complexity: O(N^2), acceptable since N (cells per notebook) is small.
    """
    n = len(arr)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions


def kendall_tau(y_true, y_pred):
    """
    Computes the Kendall Tau correlation for a single notebook.
    Formula: K = 1 - 4 * S / (n * (n - 1))

    Args:
        y_true (list): The ground truth list of cell IDs.
        y_pred (list): The predicted list of cell IDs.

    Returns:
        float: The Kendall Tau score (1.0 is perfect agreement).
    """
    n = len(y_true)
    if n <= 1:
        return 1.0

    # Map ground truth cell IDs to their correct rank (0, 1, 2...)
    rank_map = {cell_id: i for i, cell_id in enumerate(y_true)}

    # Convert prediction list to a list of ranks based on ground truth
    # Filter out any IDs in prediction that aren't in ground truth (safety check)
    pred_ranks = [rank_map[cell_id] for cell_id in y_pred if cell_id in rank_map]

    # If prediction is missing cells, the metric is ill-defined,
    # but strictly we calculate inversions on the subset or penalize.
    # Here we assume y_pred is a permutation of y_true.

    s = count_inversions(pred_ranks)

    # Calculate metric
    score = 1.0 - 4.0 * s / (n * (n - 1))
    return score


def load_processed_cells(metadata_df, split_name="train", load_cached_data=True):
    """
    Loads and processes all cells from the notebooks listed in metadata_df.
    Extracts source code, cell types, and symbolic identifiers.

    Implements deterministic caching using Parquet to save time on subsequent runs.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id', 'filepath', and optionally 'cell_order'.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: A DataFrame where each row is a cell with processed features.
    """
    cache_filename = f"{split_name}_processed.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached processed data from {cache_path}...")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {len(metadata_df)} notebooks for '{split_name}' split...")

    all_cells = []

    for _, row in metadata_df.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]

        # Determine ground truth ranks if available
        rank_map = {}
        total_cells = 0
        if "cell_order" in row and pd.notna(row["cell_order"]):
            cell_order = row["cell_order"].split()
            rank_map = {cid: i for i, cid in enumerate(cell_order)}
            total_cells = len(cell_order)

        # Read Notebook
        nb_json = read_notebook(filepath)
        if not nb_json:
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        # Iterate over all cells found in the JSON
        # Note: In test set, order in JSON is arbitrary/shuffled for markdown
        cell_ids = list(cell_types.keys())

        for cell_id in cell_ids:
            c_type = cell_types.get(cell_id, "unknown")
            c_source = sources.get(cell_id, "")

            # Feature Extraction
            c_source_clean = preprocess_text(c_source)
            # Extract identifiers (convert set to list for Parquet compatibility)
            identifiers = list(extract_identifiers(c_source))

            # Target Calculation (if training)
            rank = rank_map.get(cell_id, -1)
            if total_cells > 1:
                pct_rank = rank / (total_cells - 1)
            else:
                pct_rank = 0.0

            all_cells.append(
                {
                    "notebook_id": nb_id,
                    "cell_id": cell_id,
                    "cell_type": c_type,
                    "source": c_source,  # Original text
                    "source_clean": c_source_clean,  # Preprocessed text
                    "identifiers": identifiers,  # Symbolic features
                    "rank": rank,  # Integer rank (target)
                    "pct_rank": pct_rank,  # Normalized rank (regression target)
                }
            )

    df = pd.DataFrame(all_cells)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Saved processed data to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df
