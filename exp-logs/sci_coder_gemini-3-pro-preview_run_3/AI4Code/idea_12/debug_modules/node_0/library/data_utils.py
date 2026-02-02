import os
import json
import pandas as pd
from library.config import Config


def clean_text(text):
    """
    Cleans the cell source text by stripping leading/trailing whitespace.

    Args:
        text (str): The raw source text.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return text.strip()


def get_metadata(split):
    """
    Loads the metadata CSV for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The metadata DataFrame.
    """
    if split == "train":
        path = Config.Paths.TRAIN_METADATA
    elif split == "val":
        path = Config.Paths.VAL_METADATA
    elif split == "test":
        path = Config.Paths.TEST_METADATA
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def load_notebook_json(relative_path):
    """
    Loads the raw JSON content of a notebook.

    Args:
        relative_path (str): Path to the JSON file relative to the input directory.

    Returns:
        dict: The parsed JSON dictionary.
    """
    full_path = os.path.join(Config.Paths.INPUT_DIR, relative_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_notebook_cells(notebook_id, relative_path):
    """
    Extracts code and markdown cells from a notebook JSON.

    Args:
        notebook_id (str): The notebook ID.
        relative_path (str): Path to the JSON file relative to the input directory.

    Returns:
        dict: A dictionary containing:
            - 'id': The notebook ID.
            - 'code_cells': List of dicts {'id': cell_id, 'source': text}.
            - 'markdown_cells': List of dicts {'id': cell_id, 'source': text}.
    """
    data = load_notebook_json(relative_path)

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    code_cells = []
    markdown_cells = []

    # Iterate through the cells.
    # In the training set, code cells appear first in the JSON in their correct relative order,
    # followed by shuffled markdown cells. We preserve this order for code cells to serve as anchors.
    for cell_id, c_type in cell_types.items():
        source_text = sources.get(cell_id, "")
        cleaned_source = clean_text(source_text)

        cell_obj = {"id": cell_id, "source": cleaned_source}

        if c_type == "code":
            code_cells.append(cell_obj)
        elif c_type == "markdown":
            markdown_cells.append(cell_obj)

    return {
        "id": notebook_id,
        "code_cells": code_cells,
        "markdown_cells": markdown_cells,
    }
