import os
import json
import pandas as pd
import numpy as np
from library.config import Config


def load_metadata(split: str) -> pd.DataFrame:
    """
    Load the metadata CSV for the specified split.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split argument: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def _process_notebook(row: pd.Series, input_dir: str, is_labeled: bool) -> list:
    """
    Helper function to parse a single notebook JSON and extract cell data.

    Args:
        row: A row from the metadata DataFrame containing 'id' and 'filepath'.
        input_dir: The root directory for input files.
        is_labeled: Boolean indicating if the data is labeled (train/val) or not (test).

    Returns:
        list: A list of dictionaries, each representing a cell.
    """
    nb_id = row["id"]
    rel_path = row["filepath"]
    full_path = os.path.join(input_dir, rel_path)

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # In case of missing or corrupt file, return empty list
        return []

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    cells = []

    if is_labeled:
        # For train/val, use the ground truth 'cell_order' to determine sequence and rank
        if pd.isna(row.get("cell_order")):
            return []

        order_list = row["cell_order"].split()
        total_cells = len(order_list)

        for rank, cell_id in enumerate(order_list):
            c_type = cell_types.get(cell_id, "unknown")
            c_source = sources.get(cell_id, "")

            # Source is often a list of strings in the JSON
            if isinstance(c_source, list):
                c_source = "".join(c_source)

            # Calculate normalized rank (0.0 to 1.0)
            # If there is only 1 cell, normalized rank is 0.0
            norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

            cells.append(
                {
                    "id": nb_id,
                    "cell_id": cell_id,
                    "cell_type": c_type,
                    "source": c_source,
                    "rank": rank,
                    "norm_rank": norm_rank,
                    "ancestor_id": row.get("ancestor_id", nb_id),
                    "parent_id": row.get("parent_id", np.nan),
                }
            )
    else:
        # For test, we do not have an order. Iterate over all cells found in the JSON.
        # Order in the JSON dictionary is not guaranteed to be meaningful for the task,
        # but we just need to extract the content.
        for cell_id, c_type in cell_types.items():
            c_source = sources.get(cell_id, "")

            if isinstance(c_source, list):
                c_source = "".join(c_source)

            cells.append(
                {
                    "id": nb_id,
                    "cell_id": cell_id,
                    "cell_type": c_type,
                    "source": c_source,
                    "rank": -1,  # Placeholder
                    "norm_rank": -1.0,  # Placeholder
                    "ancestor_id": np.nan,
                    "parent_id": np.nan,
                }
            )

    return cells


def load_notebooks(
    split: str, load_cached_data: bool = True, debug: bool = False
) -> pd.DataFrame:
    """
    Load notebook data into a flat DataFrame containing cell-level information.
    Implements caching using Parquet files.

    Args:
        split: One of 'train', 'val', or 'test'.
        load_cached_data: If True, attempts to load from existing Parquet cache.
        debug: If True, loads a small subset of the data for debugging purposes.

    Returns:
        pd.DataFrame: A flat DataFrame where each row is a cell.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filename
    suffix = "_debug" if debug else ""
    cache_filename = f"{split}{suffix}_dataframe.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data from scratch (Debug={debug})...")

    # Load metadata
    df_meta = load_metadata(split)

    # Subsample if debugging
    if debug:
        sample_size = min(len(df_meta), Config.DEBUG_SAMPLE_SIZE)
        df_meta = df_meta.sample(n=sample_size, random_state=Config.SEED).copy()
        print(f"Debug mode: Sampled {sample_size} notebooks.")

    # Process notebooks
    is_labeled = split in ["train", "val"]
    all_cells = []

    # Iterate through metadata and parse JSONs
    # Using a simple loop to ensure stability and avoid multiprocessing complexity in this module
    for _, row in df_meta.iterrows():
        notebook_cells = _process_notebook(row, Config.INPUT_DIR, is_labeled)
        all_cells.extend(notebook_cells)

    # Create DataFrame
    df = pd.DataFrame(all_cells)

    if df.empty:
        print(f"Warning: No cells found for split {split}.")
        return df

    # Optimize data types
    df["cell_type"] = df["cell_type"].astype("category")

    if is_labeled:
        df["rank"] = df["rank"].astype(int)
        df["norm_rank"] = df["norm_rank"].astype(float)

    # Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df
