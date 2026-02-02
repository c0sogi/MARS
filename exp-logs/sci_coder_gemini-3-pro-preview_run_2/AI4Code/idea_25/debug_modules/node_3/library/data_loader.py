import os
import json
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    RANDOM_STATE,
)
from library.utils import seed_everything


def _read_notebook_json(filepath):
    """
    Reads a notebook JSON file and returns the cell data.
    """
    full_path = os.path.join(INPUT_DIR, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        # Return empty structure if file read fails
        return {"cell_type": {}, "source": {}}


def _process_notebooks(metadata_df, mode="train"):
    """
    Iterates through metadata, reads JSONs, and parses cell information.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata (id, filepath, cell_order, etc.).
        mode (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: A DataFrame where each row is a cell.
    """
    cell_rows = []

    # Pre-fetch columns to avoid repetitive lookups
    ids = metadata_df["id"].values
    filepaths = metadata_df["filepath"].values

    # Ancestor ID is only present in train/val
    has_ancestor = "ancestor_id" in metadata_df.columns
    ancestors = (
        metadata_df["ancestor_id"].values if has_ancestor else [None] * len(metadata_df)
    )

    # Cell order is only present in train/val
    has_order = "cell_order" in metadata_df.columns
    orders = (
        metadata_df["cell_order"].values if has_order else [None] * len(metadata_df)
    )

    for i in range(len(metadata_df)):
        nb_id = ids[i]
        rel_path = filepaths[i]
        ancestor_id = ancestors[i]
        order_str = orders[i]

        # Read JSON content
        nb_json = _read_notebook_json(rel_path)

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        # Determine the sequence of cells to process
        if mode in ["train", "val"] and order_str is not None:
            # For training/val, we trust the ground truth order
            cell_order = order_str.split()
            total_cells = len(cell_order)

            for rank, cell_id in enumerate(cell_order):
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                # Calculate normalized rank [0, 1]
                # If only 1 cell, pct_rank is 0.0
                pct_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

                cell_rows.append(
                    {
                        "cell_id": cell_id,
                        "notebook_id": nb_id,
                        "ancestor_id": ancestor_id,
                        "cell_type": c_type,
                        "source": c_source,
                        "rank": rank,
                        "pct_rank": pct_rank,
                    }
                )
        else:
            # For test set, we don't have an order. We just load all available cells.
            # The order in the JSON dictionary keys is not guaranteed to be meaningful
            # for the task, but we process them as they appear.
            # Note: The task is to predict the order, so 'rank' is unknown.

            # We iterate over keys in cell_type (or source)
            all_cell_ids = list(cell_types.keys())

            for cell_id in all_cell_ids:
                c_type = cell_types[cell_id]
                c_source = sources.get(cell_id, "")

                cell_rows.append(
                    {
                        "cell_id": cell_id,
                        "notebook_id": nb_id,
                        "ancestor_id": ancestor_id,  # Likely None or not useful for test
                        "cell_type": c_type,
                        "source": c_source,
                        "rank": -1,  # Unknown
                        "pct_rank": -1.0,  # Unknown
                    }
                )

    return pd.DataFrame(cell_rows)


def load_data(split="train", load_cached_data=True, debug_n=None):
    """
    Main function to load data for a specific split.
    Implements caching using Parquet.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from disk.
        debug_n (int, optional): If provided, limits the input metadata to N samples for debugging.

    Returns:
        pd.DataFrame: Processed cell-level data.
    """
    seed_everything(RANDOM_STATE)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(WORKING_DIR, f"{split}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # If debugging, we might want to ignore cache if the cache is full size
        # but usually loading full cache and slicing is fast enough.
        # However, to be strict with the "compute from scratch" logic if cache is invalid:
        try:
            df = pd.read_parquet(cache_path)
            # If debug_n is requested, we slice the result based on unique notebook_ids
            if debug_n is not None:
                unique_nbs = df["notebook_id"].unique()[:debug_n]
                df = df[df["notebook_id"].isin(unique_nbs)].reset_index(drop=True)
            return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    # Identify metadata file
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = VAL_METADATA_PATH
    elif split == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Apply debug limit on metadata before processing to save time
    if debug_n is not None:
        df_meta = df_meta.iloc[:debug_n].copy()

    # Process
    df_processed = _process_notebooks(df_meta, mode=split)

    # 3. Save to cache (only if not debugging, to avoid overwriting full cache with partial data)
    # However, the requirement says "IF loading fails ... OR load_cached_data is False: Compute ... Save".
    # We should probably save the specific file name. If it's a debug run, we probably shouldn't overwrite the main cache.
    # But strictly following instructions: "Save the result to the cache directory".
    # I will save only if debug_n is None to preserve the integrity of the full dataset cache.
    if debug_n is None:
        df_processed.to_parquet(cache_path, index=False)

    return df_processed
