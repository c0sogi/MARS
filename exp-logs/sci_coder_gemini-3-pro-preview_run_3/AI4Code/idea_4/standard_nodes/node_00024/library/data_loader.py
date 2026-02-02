import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers import InputExample

from library.config import Config
from library.utils import read_notebook, preprocess_text, set_seed


def get_notebook_cells(notebook_id, file_path_rel, cell_order_str=None):
    """
    Parses a notebook JSON file and returns structured lists of code and markdown cells.

    Args:
        notebook_id (str): The notebook identifier.
        file_path_rel (str): Relative path to the JSON file.
        cell_order_str (str, optional): Space-delimited string of the correct cell order.
                                        If provided (Train/Val), cells are returned with their
                                        ground truth ranks.

    Returns:
        dict: Contains:
            - 'code_cells': List of dicts {'id', 'text', 'rank' (if known)}.
            - 'markdown_cells': List of dicts {'id', 'text', 'rank' (if known)}.
            - 'all_cells_ordered': List of all cells in ground truth order (if cell_order_str provided).
    """
    # Read raw JSON data
    data = read_notebook(file_path_rel)
    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    code_cells = []
    markdown_cells = []
    all_cells_ordered = []

    if cell_order_str:
        # --- Scenario: Training/Validation (Ground Truth Available) ---
        order_list = cell_order_str.split()

        # Create a rank map: cell_id -> ground truth index
        rank_map = {cid: i for i, cid in enumerate(order_list)}

        # Iterate through the ground truth order to preserve sequence
        for rank, cell_id in enumerate(order_list):
            # Some cells in the order list might be missing from the JSON (rare data corruption), check existence
            if cell_id in cell_types:
                c_type = cell_types[cell_id]
                raw_text = sources.get(cell_id, "")
                text = preprocess_text(raw_text)

                cell_obj = {"id": cell_id, "type": c_type, "text": text, "rank": rank}

                all_cells_ordered.append(cell_obj)

                if c_type == "code":
                    code_cells.append(cell_obj)
                elif c_type == "markdown":
                    markdown_cells.append(cell_obj)

    else:
        # --- Scenario: Inference (Test Set) ---
        # Assumption: Code cells in the JSON are in the correct relative order.
        # Markdown cells are shuffled/unordered.

        # We iterate over the dictionary. In Python 3.7+, insertion order is preserved.
        # We rely on the dataset property that code cell keys appear in order.

        current_code_rank = 0

        for cell_id, c_type in cell_types.items():
            raw_text = sources.get(cell_id, "")
            text = preprocess_text(raw_text)

            cell_obj = {"id": cell_id, "type": c_type, "text": text}

            if c_type == "code":
                # Assign relative rank to code cells to act as anchors
                cell_obj["rank"] = current_code_rank
                code_cells.append(cell_obj)
                current_code_rank += 1
            elif c_type == "markdown":
                # Markdown rank is unknown
                cell_obj["rank"] = -1
                markdown_cells.append(cell_obj)

    return {
        "code_cells": code_cells,
        "markdown_cells": markdown_cells,
        "all_cells_ordered": all_cells_ordered,
    }


def prepare_training_pairs(
    metadata_path, cache_name, load_cached_data=True, debug=False
):
    """
    Generates or loads (Markdown, Next_Code) pairs for contrastive fine-tuning.

    Args:
        metadata_path (str): Path to the metadata CSV (train.csv).
        cache_name (str): Filename for the parquet cache (without extension).
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, processes only a small subset of notebooks.

    Returns:
        pd.DataFrame: DataFrame with columns ['md_text', 'code_text'].
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached training pairs from {cache_file}...")
        return pd.read_parquet(cache_file)

    # 2. Generate data from scratch
    print(f"Generating training pairs from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks.")
        df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

    pairs_data = []

    # Iterate over notebooks
    # Note: Not using tqdm to avoid cluttering logs as per instructions
    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        cell_order = row["cell_order"]
        file_path = row["file_path"]

        # Parse notebook with ground truth order
        notebook_data = get_notebook_cells(nb_id, file_path, cell_order)
        ordered_cells = notebook_data["all_cells_ordered"]

        # Strategy: For each Markdown cell, find the nearest SUBSEQUENT Code cell.
        # We iterate backwards through the ordered cells.
        next_code_text = None

        for cell in reversed(ordered_cells):
            if cell["type"] == "code":
                next_code_text = cell["text"]
            elif cell["type"] == "markdown":
                if next_code_text is not None:
                    # Found a valid (M, C_next) pair
                    pairs_data.append(
                        {"md_text": cell["text"], "code_text": next_code_text}
                    )
                # If next_code_text is None, this markdown is at the end of the notebook
                # and has no subsequent code cell to anchor to. We skip it.

    df_pairs = pd.DataFrame(pairs_data)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Saving {len(df_pairs)} pairs to {cache_file}...")
    df_pairs.to_parquet(cache_file, index=False)

    return df_pairs


class FineTuningDataset(Dataset):
    """
    PyTorch Dataset for Sentence Transformer Fine-Tuning.
    Returns InputExample objects for MultipleNegativesRankingLoss.
    """

    def __init__(self, pairs_df):
        """
        Args:
            pairs_df (pd.DataFrame): DataFrame containing 'md_text' and 'code_text'.
        """
        self.pairs = pairs_df.reset_index(drop=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        md_text = row["md_text"]
        code_text = row["code_text"]

        # The MultipleNegativesRankingLoss expects a list of texts forming a positive pair.
        # The label is implicitly 1 (positive) for the pair, and 0 for others in the batch.
        # sentence-transformers requires InputExample objects.
        return InputExample(texts=[md_text, code_text], label=1.0)
