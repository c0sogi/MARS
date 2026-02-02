import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from library.config import Config


class ContrastiveDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning Sentence Transformers using Contrastive Loss.
    Yields InputExample objects containing (Markdown, Code) pairs.
    """

    def __init__(self, pairs_df):
        """
        Args:
            pairs_df (pd.DataFrame): DataFrame containing 'markdown' and 'code' columns.
        """
        self.pairs = pairs_df.to_dict("records")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs[idx]
        # Return InputExample for MultipleNegativesRankingLoss
        # This loss expects a list of texts [anchor, positive]
        return InputExample(texts=[row["markdown"], row["code"]])


def load_json(file_path):
    """
    Helper function to load a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_notebook_data(file_path):
    """
    Reads a notebook JSON and returns lists of code and markdown cells.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple: (code_cells, markdown_cells)
            - code_cells: List of dicts {'id': str, 'source': str}, ordered as in the file.
            - markdown_cells: List of dicts {'id': str, 'source': str}, unordered/shuffled.
    """
    try:
        data = load_json(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    code_cells = []
    markdown_cells = []

    # In Python 3.7+, dictionary insertion order is preserved.
    # The problem statement guarantees code cells are in the correct relative order in the JSON.
    for cell_id, c_type in cell_types.items():
        source = sources.get(cell_id, "")
        cell_obj = {"id": cell_id, "source": source}

        if c_type == "code":
            code_cells.append(cell_obj)
        elif c_type == "markdown":
            markdown_cells.append(cell_obj)

    return code_cells, markdown_cells


def generate_training_pairs(df, load_cached_data=True, debug=False):
    """
    Generates (Markdown, Code) pairs where the Code cell immediately follows the Markdown cell
    in the ground truth order. Used for contrastive fine-tuning.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'id', 'cell_order', and 'file_path'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug (bool): If True, processes only a small subset of data and uses a debug cache file.

    Returns:
        pd.DataFrame: DataFrame with columns ['markdown', 'code'].
    """
    # Define cache path based on debug status to prevent overwriting full data with debug data
    cache_filename = "train_pairs_debug.parquet" if debug else "train_pairs.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training pairs from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Generate from scratch
    print(f"Generating training pairs (Debug={debug})...")

    pairs = []

    # If debug, sample the dataframe
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Iterate through notebooks
    # We do not use tqdm here to avoid excessive printing as per requirements
    for _, row in df.iterrows():
        # Construct full path. Metadata paths are relative (e.g., "train/id.json")
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            data = load_json(file_path)
        except Exception:
            continue

        cell_order_str = row["cell_order"]
        if pd.isna(cell_order_str):
            continue

        cell_order = cell_order_str.split()
        sources = data.get("source", {})
        cell_types = data.get("cell_type", {})

        # We look for transitions: Markdown -> Code
        # Iterate through the ground truth order
        for i in range(len(cell_order) - 1):
            curr_id = cell_order[i]
            next_id = cell_order[i + 1]

            curr_type = cell_types.get(curr_id)
            next_type = cell_types.get(next_id)

            # Check if this is a Markdown cell followed by a Code cell
            if curr_type == "markdown" and next_type == "code":
                md_text = sources.get(curr_id, "")
                code_text = sources.get(next_id, "")

                # Basic validation and truncation to keep memory usage reasonable
                # We truncate to 2000 chars here; tokenizer will truncate further to MAX_LEN
                if md_text and code_text:
                    pairs.append({"markdown": md_text[:2000], "code": code_text[:2000]})

    pairs_df = pd.DataFrame(pairs)

    # 3. Save to cache
    print(f"Saving {len(pairs_df)} pairs to {cache_path}")
    # Ensure directory exists (handled by Config, but good practice)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pairs_df.to_parquet(cache_path, index=False)

    return pairs_df
