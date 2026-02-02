import json
import os
import pandas as pd
from library.config import Config


def load_metadata(split="train", sample_size=None):
    """
    Loads the metadata DataFrame for the specified split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        sample_size (int, optional): Number of rows to sample for debugging.
                                     If None, loads the full dataset.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_CSV
    elif split == "val":
        path = Config.VAL_CSV
    elif split == "test":
        path = Config.TEST_CSV
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    # Apply sampling if requested
    if sample_size is not None and sample_size > 0:
        if len(df) > sample_size:
            # Deterministic sampling using the seed from Config
            df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(
                drop=True
            )

    return df


def get_ordered_cells(cell_order_str):
    """
    Parses the space-delimited cell order string into a list of cell IDs.

    Args:
        cell_order_str (str): Space-delimited string of cell IDs.

    Returns:
        list: List of cell IDs in the correct order.
    """
    if not isinstance(cell_order_str, str):
        return []
    return cell_order_str.split()


def load_notebook(relative_path):
    """
    Loads a single notebook JSON file and separates its content into code and markdown cells.

    Args:
        relative_path (str): The relative file path to the notebook JSON
                             (e.g., 'train/00001756c60be8.json').

    Returns:
        dict: A dictionary containing:
            - 'code_cells': dict mapping cell_id to source text for code cells.
            - 'markdown_cells': dict mapping cell_id to source text for markdown cells.
    """
    # Construct the full path using the input directory from Config
    full_path = os.path.join(Config.INPUT_DIR, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Notebook file not found: {full_path}")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        # In case of JSON decode errors or other IO issues
        raise RuntimeError(f"Failed to load notebook {full_path}: {e}")

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    code_cells = {}
    markdown_cells = {}

    for cell_id, c_type in cell_types.items():
        # Retrieve source text, defaulting to empty string if missing
        source_text = sources.get(cell_id, "")

        if c_type == "code":
            code_cells[cell_id] = source_text
        elif c_type == "markdown":
            markdown_cells[cell_id] = source_text
        # Ignore other cell types (e.g., raw) if any

    return {"code_cells": code_cells, "markdown_cells": markdown_cells}
