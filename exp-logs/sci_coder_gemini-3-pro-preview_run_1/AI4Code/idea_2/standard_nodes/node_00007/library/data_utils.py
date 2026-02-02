import json
import os
from library.config import Config


def read_notebook(filepath):
    """
    Reads a notebook JSON file from the input directory.

    Args:
        filepath (str): The relative path (e.g., 'train/00001756c60be8.json')
                        or absolute path to the notebook file.

    Returns:
        dict: The parsed JSON content of the notebook.
    """
    # Determine the full path
    if os.path.isabs(filepath):
        full_path = filepath
    else:
        full_path = os.path.join(Config.input_dir, filepath)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Notebook file not found at: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def preprocess_text(text):
    """
    Cleans and truncates source text for tokenization preparation.

    Args:
        text (str): The raw source code or markdown text.

    Returns:
        str: The cleaned and truncated text.
    """
    if not isinstance(text, str):
        return ""

    # Strip leading/trailing whitespace
    text = text.strip()

    # Truncate to a reasonable character limit.
    # The Sentence Transformer uses a max_length of 128 tokens (approx 500-600 chars).
    # We keep 1000 characters to ensure we capture sufficient context before tokenization
    # while preventing memory issues with extremely large cells.
    return text[:1000]


def get_ground_truth_ranks(cell_order_str):
    """
    Parses the ground truth cell order string into a dictionary mapping cell IDs to ranks.

    Args:
        cell_order_str (str): A space-delimited string of cell IDs (e.g., "id1 id2 id3").

    Returns:
        dict: A dictionary where keys are cell_ids and values are their 0-indexed integer rank.
    """
    if not cell_order_str:
        return {}

    order_list = cell_order_str.split()
    return {cell_id: rank for rank, cell_id in enumerate(order_list)}


def load_notebook_cells(notebook_id, filepath, cell_order_str=None):
    """
    Helper function to load a notebook and flatten it into a list of cell dictionaries.
    This serves as a bridge between raw data loading and feature extraction.

    Args:
        notebook_id (str): The unique identifier for the notebook.
        filepath (str): The path to the notebook JSON file.
        cell_order_str (str, optional): The ground truth cell order string. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries, each representing a cell with keys:
                    'notebook_id', 'cell_id', 'cell_type', 'source', and optionally 'rank'.
    """
    # Load raw notebook content
    nb_json = read_notebook(filepath)

    cell_types = nb_json.get("cell_type", {})
    sources = nb_json.get("source", {})

    # Parse ground truth ranks if available
    ranks = {}
    if cell_order_str:
        ranks = get_ground_truth_ranks(cell_order_str)
        total_cells = len(ranks)

    cells = []

    # Iterate through all cells found in the notebook
    for cell_id, c_type in cell_types.items():
        # Retrieve and clean source text
        raw_source = sources.get(cell_id, "")
        clean_source = preprocess_text(raw_source)

        cell_data = {
            "notebook_id": notebook_id,
            "cell_id": cell_id,
            "cell_type": c_type,
            "source": clean_source,
        }

        # Attach rank information if ground truth is provided
        if cell_order_str:
            if cell_id in ranks:
                rank = ranks[cell_id]
                cell_data["rank"] = rank
                # Calculate relative rank (0.0 to 1.0) which is often used as a regression target
                cell_data["rel_rank"] = (
                    rank / (total_cells - 1) if total_cells > 1 else 0.5
                )
            else:
                # Handle edge case where a cell in JSON is not in the order string
                cell_data["rank"] = -1
                cell_data["rel_rank"] = -1.0

        cells.append(cell_data)

    return cells
