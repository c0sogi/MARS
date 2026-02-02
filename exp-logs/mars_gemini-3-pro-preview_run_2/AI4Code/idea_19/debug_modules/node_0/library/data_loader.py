import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook_json, preprocess_text


def process_notebooks(df_meta, split):
    """
    Parses notebook JSON files and constructs a DataFrame of cells.

    Args:
        df_meta (pd.DataFrame): Metadata containing notebook IDs and filepaths.
        split (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: Processed cell-level data.
    """
    data = []

    # Pre-fetch column indices for faster access
    id_idx = df_meta.columns.get_loc("id")
    path_idx = df_meta.columns.get_loc("filepath")

    has_orders = "cell_order" in df_meta.columns
    order_idx = df_meta.columns.get_loc("cell_order") if has_orders else -1

    has_ancestors = "ancestor_id" in df_meta.columns
    ancestor_idx = df_meta.columns.get_loc("ancestor_id") if has_ancestors else -1

    # Iterate over metadata
    for row in df_meta.itertuples(index=False):
        nb_id = row[id_idx]
        rel_path = row[path_idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load JSON
        nb_json = read_notebook_json(full_path)
        if not nb_json:
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        # Determine cell processing order
        if split in ["train", "val"] and has_orders:
            # Use ground truth order
            cell_order = row[order_idx].split()
        else:
            # Use JSON key order (Code cells are first and ordered, MD are shuffled after)
            cell_order = list(cell_types.keys())

        total_cells = len(cell_order)
        ancestor = row[ancestor_idx] if has_ancestors else nb_id

        for rank, cell_id in enumerate(cell_order):
            c_type = cell_types.get(cell_id, "code")  # Default to code if missing
            c_source = sources.get(cell_id, "")

            # Clean source text
            clean_source = preprocess_text(c_source)

            # Calculate targets
            if split in ["train", "val"]:
                # Normalized rank: 0.0 to 1.0
                pct_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0
                current_rank = rank
            else:
                pct_rank = np.nan
                current_rank = np.nan

            data.append(
                {
                    "id": nb_id,
                    "cell_id": cell_id,
                    "cell_type": c_type,
                    "source": clean_source,
                    "rank": current_rank,
                    "pct_rank": pct_rank,
                    "ancestor_id": ancestor,
                }
            )

    df = pd.DataFrame(data)

    # Optimize dtypes
    if not df.empty:
        df["cell_type"] = df["cell_type"].astype("category")
        df["id"] = df["id"].astype("category")
        if "ancestor_id" in df.columns:
            df["ancestor_id"] = df["ancestor_id"].astype("category")

    return df


def load_data(split="train", load_cached_data=True):
    """
    Main entry point to load data for a specific split.
    Handles caching and debug sampling.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame: The cell-level dataset.
    """
    # Determine cache path based on split
    if split == "train":
        cache_path = Config.CACHE_TRAIN_DATAFRAME
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        cache_path = Config.CACHE_VAL_DATAFRAME
        meta_path = Config.VAL_METADATA_PATH
    elif split == "test":
        cache_path = Config.CACHE_TEST_DATAFRAME
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            # If in debug mode, we might want to sample the cached data
            # or strictly enforce reprocessing.
            # Usually, if cache exists, we assume it matches the intent.
            # However, if Config.DEBUG is True but cache is full size, we should probably slice it.
            if Config.DEBUG:
                # Get unique notebooks and sample
                unique_ids = df["id"].unique()
                if len(unique_ids) > Config.DEBUG_SAMPLE_SIZE:
                    print(f"DEBUG: Subsampling cached {split} data...")
                    sample_ids = unique_ids[: Config.DEBUG_SAMPLE_SIZE]
                    df = df[df["id"].isin(sample_ids)].reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {split} data from raw files...")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Apply Debug Sampling to Metadata BEFORE processing (saves time)
    if Config.DEBUG:
        print(f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks from metadata.")
        if len(df_meta) > Config.DEBUG_SAMPLE_SIZE:
            df_meta = df_meta.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    df_processed = process_notebooks(df_meta, split)

    # 3. Save to cache (only if not in debug mode to avoid overwriting full cache with partial data,
    #    OR if we decide debug runs utilize their own cache.
    #    Given instructions: 'Save the result to the cache directory... for future runs'
    #    We will save it. If user runs debug then full, they should probably clear cache or set load_cached_data=False)
    print(f"Saving {split} data to cache: {cache_path}")
    df_processed.to_parquet(cache_path, index=False)

    return df_processed
