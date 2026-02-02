import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import config
from library.utils import read_notebook, preprocess_text


class NotebookLoader:
    """
    Iterator class to load notebooks based on a metadata CSV file.
    """

    def __init__(self, metadata_path, input_dir=config.INPUT_DIR):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            input_dir (str): Root directory containing the notebook files.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.input_dir = input_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (notebook_id, json_data, cell_order)
            - notebook_id (str): ID of the notebook.
            - json_data (dict): Parsed JSON content of the notebook.
            - cell_order (list): List of cell IDs in correct order (or None if not available).
        """
        row = self.metadata.iloc[idx]
        notebook_id = row["id"]

        # Construct full file path
        # row['file_path'] is relative, e.g., "train/xxxxx.json"
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load JSON
        data = read_notebook(file_path)

        # Get cell order (ground truth if available)
        cell_order = None
        if "cell_order" in row and pd.notna(row["cell_order"]):
            cell_order = str(row["cell_order"]).split()

        return notebook_id, data, cell_order

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def load_contrastive_pairs(metadata_path, cache_name, load_cached_data=True):
    """
    Generates or loads pairs of (Markdown, Next_Code) for contrastive learning.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_name (str): Base name of the cache file (e.g., 'train_pairs').
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'markdown' and 'code' columns.
    """
    # Adjust cache name for debug mode to avoid overwriting full data cache
    if config.DEBUG_SAMPLE_SIZE is not None:
        cache_name = f"{cache_name}_debug{config.DEBUG_SAMPLE_SIZE}"

    # Ensure extension is parquet
    if not cache_name.endswith(".parquet"):
        cache_name += ".parquet"

    cache_path = os.path.join(config.WORKING_DIR, cache_name)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading contrastive pairs from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from scratch...")

    # 2. Compute from Scratch
    print(f"Generating contrastive pairs from {metadata_path}...")

    loader = NotebookLoader(metadata_path)
    pairs = []

    # Determine range of notebooks to process
    total_notebooks = len(loader)
    if config.DEBUG_SAMPLE_SIZE is not None:
        limit = min(total_notebooks, config.DEBUG_SAMPLE_SIZE)
        print(f"Debug mode: Sampling first {limit} notebooks out of {total_notebooks}.")
        indices = range(limit)
    else:
        indices = range(total_notebooks)

    for idx in indices:
        notebook_id, data, cell_order = loader[idx]

        # Skip if data load failed or no ground truth order
        if data is None or cell_order is None:
            continue

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        # Iterate through the order to find MD -> Code transitions
        # cell_order is a list of cell_ids
        for i in range(len(cell_order) - 1):
            curr_id = cell_order[i]
            next_id = cell_order[i + 1]

            # Check types safely
            curr_type = cell_types.get(curr_id, "")
            next_type = cell_types.get(next_id, "")

            if curr_type == "markdown" and next_type == "code":
                md_text = sources.get(curr_id, "")
                code_text = sources.get(next_id, "")

                # Preprocess text (truncate and clean)
                md_text = preprocess_text(md_text)
                code_text = preprocess_text(code_text)

                # Only add if both contain content
                if md_text and code_text:
                    pairs.append({"markdown": md_text, "code": code_text})

    df = pd.DataFrame(pairs)

    # 3. Save to Cache
    print(f"Saving {len(df)} pairs to cache: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class ContrastiveDataset(Dataset):
    """
    PyTorch Dataset for Contrastive Learning.
    Yields (anchor, positive) pairs: (Markdown, Next Code).
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
        row = self.pairs.iloc[idx]
        return {"markdown": row["markdown"], "code": row["code"]}
