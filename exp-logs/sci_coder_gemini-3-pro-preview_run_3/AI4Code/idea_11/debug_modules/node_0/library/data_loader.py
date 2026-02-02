import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import read_notebook, preprocess_text


class FineTuningDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning the backbone model using contrastive learning.

    Attributes:
        pairs (pd.DataFrame): DataFrame containing 'markdown' and 'code' text columns.
    """

    def __init__(self, pairs_df):
        """
        Args:
            pairs_df (pd.DataFrame): DataFrame with 'markdown' and 'code' columns.
        """
        self.pairs = pairs_df.reset_index(drop=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (markdown_text, code_text)
        """
        row = self.pairs.iloc[idx]
        return row["markdown"], row["code"]


class NotebookTextLoader:
    """
    Helper class to load notebook texts for feature extraction.
    It abstracts the reading of JSON files and separation of cell types.
    """

    def __init__(self, metadata_path):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train/val/test).
        """
        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the notebook data at the specified index.

        Returns:
            dict: {
                'id': str,
                'code_cells': list of tuples (cell_id, source_text),
                'markdown_cells': list of tuples (cell_id, source_text),
                'cell_order': list of str (ground truth IDs) or None
            }
        """
        row = self.df.iloc[idx]
        notebook_id = row["id"]
        # Construct full path using Config root and relative path from metadata
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Read notebook content
        # Returns lists of (cell_id, source_text)
        try:
            code_cells, markdown_cells = read_notebook(file_path)
        except Exception:
            # Fallback for corrupted or missing files (though metadata check passed)
            code_cells, markdown_cells = [], []

        cell_order = None
        if "cell_order" in row and pd.notna(row["cell_order"]):
            cell_order = row["cell_order"].split()

        return {
            "id": notebook_id,
            "code_cells": code_cells,
            "markdown_cells": markdown_cells,
            "cell_order": cell_order,
        }


def prepare_relaxed_pairs(metadata_path, sample_size=None, load_cached_data=True):
    """
    Generates (markdown, nearest_subsequent_code) pairs for contrastive learning.

    Implements the 'Relaxed Proximity' strategy:
    1. Reconstructs the correct order of cells.
    2. Iterates backwards to find the nearest subsequent code cell for each markdown cell.
    3. Caches the result to a Parquet file for efficiency.

    Args:
        metadata_path (str): Path to the metadata CSV (e.g., Config.TRAIN_PATH).
        sample_size (int, optional): Number of notebooks to sample. Defaults to None (all).
        load_cached_data (bool): Whether to attempt loading from cache. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with columns ['markdown', 'code'].
    """
    cache_path = Config.TRAIN_PAIRS_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading training pairs from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating training pairs from scratch...")

    # 2. Load metadata
    df = pd.read_csv(metadata_path)

    # 3. Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)

    pairs_list = []

    # 4. Process notebooks
    for _, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        try:
            code_cells, md_cells = read_notebook(file_path)
        except Exception:
            continue

        if not code_cells or not md_cells:
            continue

        # Ground truth order is required for pairing
        if "cell_order" not in row or pd.isna(row["cell_order"]):
            continue

        cell_order = row["cell_order"].split()

        # Create lookups for valid cells
        valid_code_cells = {cid: txt for cid, txt in code_cells}
        valid_md_cells = {cid: txt for cid, txt in md_cells}

        # Reconstruct the sequence of content types and texts
        # We only care about cells that exist in both the file and the order list
        ordered_content = []
        for cid in cell_order:
            if cid in valid_code_cells:
                ordered_content.append(("c", valid_code_cells[cid]))
            elif cid in valid_md_cells:
                ordered_content.append(("m", valid_md_cells[cid]))

        # Find pairs: Markdown -> Nearest Subsequent Code
        # We iterate backwards. The 'last seen' code cell is the nearest subsequent one.
        next_code_text = None

        for c_type, text in reversed(ordered_content):
            if c_type == "c":
                next_code_text = text
            elif c_type == "m":
                if next_code_text is not None:
                    # Found a pair
                    pairs_list.append(
                        {
                            "markdown": preprocess_text(text),
                            "code": preprocess_text(next_code_text),
                        }
                    )

    pairs_df = pd.DataFrame(pairs_list)

    # 5. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pairs_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(pairs_df)} pairs to cache: {cache_path}")

    return pairs_df
