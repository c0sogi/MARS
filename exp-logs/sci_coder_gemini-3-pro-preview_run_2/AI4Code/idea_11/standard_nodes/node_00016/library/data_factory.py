import os
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library import config, utils

# ------------------------------------------------------------------------------
# Helper Functions (Picklable for Parallel Execution)
# ------------------------------------------------------------------------------


def _process_single_notebook(row, data_dir, mode):
    """
    Processes a single notebook to extract cells and compute targets.

    Args:
        row (pd.Series): Metadata row containing 'id', 'filepath', 'cell_order', etc.
        data_dir (str): Base directory for input data.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        list: A list of dictionaries, each representing a cell.
    """
    notebook_id = row["id"]
    rel_path = row["filepath"]
    full_path = os.path.join(data_dir, rel_path)

    # Read JSON content
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)
    except Exception:
        # Return empty if file read fails
        return []

    # Extract source and cell_type
    # Handle cases where source is a list of strings or a single string
    sources = nb_json.get("source", {})
    cell_types = nb_json.get("cell_type", {})

    # Determine the list of cells to process
    if mode in ["train", "val"]:
        # Use the ground truth cell_order
        if pd.isna(row.get("cell_order")):
            return []
        cell_order = row["cell_order"].split()
    else:
        # For test, we process all cells present in the JSON
        # We rely on the keys in cell_type or source
        cell_order = list(cell_types.keys())

    # Pre-calculate Code Skeleton for Target Generation (Train/Val only)
    if mode in ["train", "val"]:
        code_cells = [cid for cid in cell_order if cell_types.get(cid, "") == "code"]
        num_code_cells = len(code_cells)

        # Mapping of code cell ID to its integer rank (0, 1, 2...)
        code_rank_map = {cid: i for i, cid in enumerate(code_cells)}

        # Identify gaps between code cells to distribute markdown cells evenly
        # Gaps: Before C0, Between C0-C1, ..., After Cn
        # We need to know which gap a markdown cell belongs to and its position in that gap

        # Pass 1: Build gap structure
        # gap_index -> list of markdown cell_ids
        # gap_index 0 is before first code cell
        # gap_index k is after code cell (k-1)
        gaps = {i: [] for i in range(num_code_cells + 1)}
        current_gap_idx = 0

        for cid in cell_order:
            ctype = cell_types.get(cid, "unknown")
            if ctype == "code":
                current_gap_idx += 1
            elif ctype == "markdown":
                gaps[current_gap_idx].append(cid)

        # Calculate local ranks within gaps
        # cell_id -> local_rank (0 to 1)
        md_skeleton_ranks = {}
        for gap_idx, md_list in gaps.items():
            count = len(md_list)
            if count == 0:
                continue
            for i, md_id in enumerate(md_list):
                # Distribute evenly: (i + 1) / (count + 1)
                # Gap 0 (before C0): relative position -1 + offset
                # Gap k (after C(k-1)): relative position (k-1) + offset
                offset = (i + 1) / (count + 1)
                skeleton_pos = (gap_idx - 1) + offset
                md_skeleton_ranks[md_id] = skeleton_pos

    processed_cells = []

    # Iterate to build final rows
    for cid in cell_order:
        ctype = cell_types.get(cid, "unknown")
        source_content = sources.get(cid, "")

        # Normalize source to string
        if isinstance(source_content, list):
            source_content = "".join(source_content)

        cell_data = {
            "id": notebook_id,
            "cell_id": cid,
            "cell_type": ctype,
            "source": source_content,
            "ancestor_id": row.get(
                "ancestor_id", notebook_id
            ),  # Default to self if missing
        }

        if mode in ["train", "val"]:
            if ctype == "code":
                # Code cells don't have a regression target in this formulation,
                # but we store their integer rank for reference if needed.
                # We use -1.0 or NaN to indicate they are anchors.
                cell_data["rank"] = float(code_rank_map.get(cid, -1))
                cell_data["is_code"] = 1
            else:
                # Markdown Target Generation
                if num_code_cells > 0:
                    # Skeleton Rank normalized by number of code cells
                    # Range roughly [-1/N, 1 + 1/N] -> usually we want [0, 1]
                    # We simply divide by num_code_cells.
                    # At inference, we place Code Cell i at i/num_code_cells.
                    raw_rank = md_skeleton_ranks.get(cid, 0.0)
                    cell_data["rank"] = raw_rank / num_code_cells
                else:
                    # Fallback for notebooks with NO code cells: Global Normalized Rank
                    # Just use position in the list
                    total_cells = len(cell_order)
                    current_idx = cell_order.index(cid)
                    cell_data["rank"] = (
                        current_idx / (total_cells - 1) if total_cells > 1 else 0.5
                    )

                cell_data["is_code"] = 0
        else:
            # Test mode: No rank
            cell_data["rank"] = np.nan
            cell_data["is_code"] = 1 if ctype == "code" else 0

        processed_cells.append(cell_data)

    return processed_cells


def _load_dataset(metadata_path, data_dir, cache_path, load_cached_data, mode):
    """
    Core function to load, process, and cache dataset.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        utils.log_message(f"Loading cached {mode} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            utils.log_message(f"Successfully loaded {len(df)} cells.")
            return df
        except Exception as e:
            utils.log_message(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    utils.log_message(f"Processing {len(df_meta)} notebooks for {mode}...")

    # 3. Parallel Processing
    # We process notebooks in parallel batches
    results = Parallel(n_jobs=config.NUM_JOBS, backend="loky")(
        delayed(_process_single_notebook)(row, config.INPUT_DIR, mode)
        for _, row in df_meta.iterrows()
    )

    # Flatten results
    all_cells = [cell for nb_cells in results for cell in nb_cells]

    # Create DataFrame
    df = pd.DataFrame(all_cells)

    # Optimize types
    df["cell_type"] = df["cell_type"].astype("category")
    df["id"] = df["id"].astype("category")
    df["ancestor_id"] = df["ancestor_id"].astype("category")

    # 4. Save to Cache
    utils.log_message(f"Saving processed data to {cache_path}...")
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        utils.log_message(f"Warning: Failed to save cache: {e}")

    utils.log_message(f"Finished processing. Total cells: {len(df)}")
    return df


# ------------------------------------------------------------------------------
# Public Interface
# ------------------------------------------------------------------------------


def load_train_data(load_cached_data=True):
    """
    Loads and processes the training set.
    """
    return _load_dataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        data_dir=config.TRAIN_DIR,
        cache_path=config.CACHE_TRAIN_DATAFRAME,
        load_cached_data=load_cached_data,
        mode="train",
    )


def load_val_data(load_cached_data=True):
    """
    Loads and processes the validation set.
    """
    return _load_dataset(
        metadata_path=config.VAL_METADATA_PATH,
        data_dir=config.TRAIN_DIR,
        cache_path=config.CACHE_VAL_DATAFRAME,
        load_cached_data=load_cached_data,
        mode="val",
    )


def load_test_data(load_cached_data=True):
    """
    Loads and processes the test set.
    """
    return _load_dataset(
        metadata_path=config.TEST_METADATA_PATH,
        data_dir=config.TEST_DIR,
        cache_path=config.CACHE_TEST_DATAFRAME,
        load_cached_data=load_cached_data,
        mode="test",
    )
