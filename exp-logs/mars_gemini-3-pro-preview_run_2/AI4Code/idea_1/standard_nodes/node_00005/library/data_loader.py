import os
import json
import pandas as pd
import numpy as np
from library.config import *


def load_notebook_json(filepath):
    """
    Reads a notebook JSON file.
    """
    full_path = os.path.join(INPUT_DIR, filepath)
    if not os.path.exists(full_path):
        return {}

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_regression_data(data_type="train", load_cached_data=True, max_samples=None):
    """
    Generates or loads the dataset for the regression model (Markdown Text -> Rank).

    Args:
        data_type (str): 'train' or 'val'.
        load_cached_data (bool): Whether to load from parquet cache if available.
        max_samples (int, optional): Limit the number of notebooks processed for debugging.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'cell_id', 'text', 'rank'].
    """
    # Determine paths
    if data_type == "train":
        metadata_path = TRAIN_METADATA_PATH
        cache_filename = "train_regression_data_v2.parquet"
    elif data_type == "val":
        metadata_path = VAL_METADATA_PATH
        cache_filename = "val_regression_data_v2.parquet"
    else:
        raise ValueError("data_type must be 'train' or 'val'")

    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if max_samples is not None:
                return df.iloc[
                    :max_samples
                ]  # Approximate subsampling on rows, not notebooks
            return df
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    if max_samples is not None:
        df_meta = df_meta.iloc[:max_samples]

    data_rows = []

    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]
        cell_order_str = row["cell_order"]

        if pd.isna(cell_order_str):
            continue

        correct_order = cell_order_str.split()
        total_cells = len(correct_order)

        # If notebook is too small to have meaningful rank, skip or handle
        if total_cells < 2:
            continue

        nb_json = load_notebook_json(filepath)
        if not nb_json:
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        # Create a map of cell_id -> rank
        rank_map = {cid: i for i, cid in enumerate(correct_order)}

        # Iterate through correct order to extract markdown cells
        for cell_id in correct_order:
            ctype = cell_types.get(cell_id, "unknown")

            if ctype == "markdown":
                source_text = sources.get(cell_id, "")
                rank = rank_map[cell_id]

                # Normalized rank: 0.0 (top) to 1.0 (bottom)
                norm_rank = rank / (total_cells - 1)

                data_rows.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "text": source_text,
                        "rank": norm_rank,
                    }
                )

    df_result = pd.DataFrame(data_rows)

    # 3. Save to cache
    try:
        df_result.to_parquet(cache_path, index=False)
    except Exception:
        pass  # Non-critical failure

    return df_result


def get_inference_data(data_type="test", max_samples=None):
    """
    Loads notebooks for inference.
    Returns a structure containing ordered code cells and unordered markdown cells.

    Args:
        data_type (str): 'test' or 'val'.
        max_samples (int, optional): Limit number of notebooks.

    Returns:
        list of dict: Each dict contains:
            - 'id': notebook id
            - 'code_cells': list of code cell IDs (in relative order)
            - 'markdown_cells': list of (cell_id, text) tuples
    """
    if data_type == "test":
        metadata_path = TEST_METADATA_PATH
    elif data_type == "val":
        metadata_path = VAL_METADATA_PATH
    else:
        raise ValueError("data_type must be 'test' or 'val'")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    if max_samples is not None:
        df_meta = df_meta.iloc[:max_samples]

    inference_data = []

    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]

        nb_json = load_notebook_json(filepath)
        if not nb_json:
            continue

        cell_types = nb_json.get("cell_type", {})
        sources = nb_json.get("source", {})

        # According to the task description:
        # "The code cells are in their original (correct) order.
        #  The markdown cells have been shuffled and placed after the code cells."
        # This implies we can iterate the keys in the JSON (which preserve insertion order in Python 3.7+)
        # and separating them by type will yield:
        # 1. Code cells in correct relative order.
        # 2. Markdown cells in shuffled order.

        code_cells = []
        markdown_cells = []

        # We iterate over the keys of cell_type or source.
        # Usually source and cell_type have the same keys in the same order.
        all_cell_ids = list(cell_types.keys())

        for cell_id in all_cell_ids:
            ctype = cell_types.get(cell_id, "unknown")
            source_text = sources.get(cell_id, "")

            if ctype == "code":
                code_cells.append(cell_id)
            elif ctype == "markdown":
                markdown_cells.append((cell_id, source_text))

        inference_data.append(
            {"id": nb_id, "code_cells": code_cells, "markdown_cells": markdown_cells}
        )

    return inference_data
