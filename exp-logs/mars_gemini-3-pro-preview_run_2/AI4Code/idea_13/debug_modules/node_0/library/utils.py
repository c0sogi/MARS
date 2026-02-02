import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].
    This corresponds to the number of swaps required to sort the array.

    Args:
        a (list[int]): List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] > a[j]:
                inversions += 1
    return inversions


def kendall_tau_metric(df_preds, df_truth):
    """
    Calculates the Kendall Tau correlation metric as defined in the competition.
    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        df_preds (pd.DataFrame): DataFrame with columns ['id', 'cell_order'].
        df_truth (pd.DataFrame): DataFrame with columns ['id', 'cell_order'].

    Returns:
        float: The Kendall Tau score.
    """
    # Create lookup dictionaries
    preds_dict = dict(zip(df_preds["id"], df_preds["cell_order"]))
    truth_dict = dict(zip(df_truth["id"], df_truth["cell_order"]))

    total_swaps = 0
    total_possible_pairs = 0  # Sum of n*(n-1)

    # Validate on the intersection of IDs, or strictly on truth IDs
    valid_ids = df_truth["id"].unique()

    for nb_id in valid_ids:
        if nb_id not in preds_dict:
            continue

        truth_order = truth_dict[nb_id].split()
        pred_order = preds_dict[nb_id].split()

        n = len(truth_order)
        if n <= 1:
            continue

        # Map cell IDs to their correct ground truth rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(truth_order)}

        # Convert predicted order to a list of ranks
        # We filter to ensure we only consider cells present in the ground truth
        pred_ranks = [rank_map[cid] for cid in pred_order if cid in rank_map]

        # Calculate swaps (inversions) for this notebook
        swaps = count_inversions(pred_ranks)

        total_swaps += swaps
        total_possible_pairs += n * (n - 1)

    if total_possible_pairs == 0:
        return 1.0

    return 1 - 4 * (total_swaps / total_possible_pairs)


def read_notebook(filepath):
    """
    Reads a JSON notebook file.

    Args:
        filepath (str): Path to the .json file.

    Returns:
        dict: The parsed JSON content, or None if reading fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def preprocess_data(
    metadata_path, output_path, load_cached_data=True, debug=False, debug_size=None
):
    """
    Loads notebook data based on metadata, parses JSONs, and creates a flat DataFrame.
    Implements caching to Parquet files to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV.
        output_path (str): Path to save/load the processed parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): Whether to run in debug mode (sample data).
        debug_size (int): Number of samples for debug mode.

    Returns:
        pd.DataFrame: Processed dataframe with columns:
                      ['id', 'cell_id', 'cell_type', 'source', 'rank', 'norm_rank', 'ancestor_id', 'parent_id']
    """
    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(output_path):
        try:
            print(f"Loading cached data from {output_path}...")
            df = pd.read_parquet(output_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Metadata
    print(f"Processing data from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    if debug and debug_size is not None:
        df_meta = df_meta.iloc[:debug_size].copy()
        print(f"Debug mode: sampled {len(df_meta)} notebooks.")

    # 3. Process Notebooks
    rows = []
    input_dir = Config.INPUT_DIR

    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        rel_path = row["filepath"]
        full_path = os.path.join(input_dir, rel_path)

        nb_json = read_notebook(full_path)
        if nb_json is None:
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        # Determine cell IDs and Ranks
        if "cell_order" in row and pd.notna(row["cell_order"]):
            # Train/Val case: Order is known
            cell_order = row["cell_order"].split()
            rank_map = {cid: i for i, cid in enumerate(cell_order)}
            cell_ids = cell_order
        else:
            # Test case: Order is unknown, use keys from JSON
            cell_ids = list(cell_types.keys())
            rank_map = {}

        total_cells = len(cell_ids)
        ancestor_id = row.get("ancestor_id", np.nan)
        parent_id = row.get("parent_id", np.nan)

        for cid in cell_ids:
            rank = rank_map.get(cid, np.nan)

            # Calculate normalized rank (0.0 to 1.0)
            if not np.isnan(rank) and total_cells > 1:
                norm_rank = rank / (total_cells - 1)
            elif not np.isnan(rank):
                norm_rank = 0.0
            else:
                norm_rank = np.nan

            rows.append(
                {
                    "id": nb_id,
                    "cell_id": cid,
                    "cell_type": cell_types.get(cid, "unknown"),
                    "source": sources.get(cid, ""),
                    "rank": rank,
                    "norm_rank": norm_rank,
                    "ancestor_id": ancestor_id,
                    "parent_id": parent_id,
                    "n_cells": total_cells,
                }
            )

    df_processed = pd.DataFrame(rows)

    # 4. Save to Cache
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_parquet(output_path, index=False)
    print(f"Saved processed data to {output_path}")

    return df_processed
