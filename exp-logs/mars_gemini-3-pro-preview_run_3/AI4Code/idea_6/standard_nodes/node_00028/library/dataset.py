import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook, set_seed


class NotebookLoader:
    """
    Helper class to load notebooks based on metadata.
    Provides access to notebook content and metadata rows.
    """

    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            notebook_data = read_notebook(row["file_path"])
        except Exception:
            notebook_data = {}
        return row, notebook_data


def create_relaxed_proximity_pairs(
    metadata_path, mode="train", debug=False, load_cached_data=True
):
    """
    Generates (markdown, code) pairs for contrastive fine-tuning (Stage 1).
    Implements 'Relaxed Proximity Pairing': pairs every markdown cell with the
    nearest subsequent code cell. If a markdown cell is at the end of the notebook,
    it is paired with the last available code cell.

    Args:
        metadata_path (str): Path to the metadata CSV.
        mode (str): 'train' or 'val', used for cache naming.
        debug (bool): If True, process only a small subset.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: DataFrame containing 'markdown_text' and 'code_text'.
    """
    cache_filename = f"pairs_{mode}{'_debug' if debug else ''}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pairs from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing pairs for {mode} (debug={debug})...")

    loader = NotebookLoader(metadata_path)
    df = loader.df

    if debug:
        df = df.head(100)

    pairs_data = []

    for _, row in df.iterrows():
        try:
            data = read_notebook(row["file_path"])
        except Exception:
            continue

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        # Get ground truth order
        if "cell_order" not in row or pd.isna(row["cell_order"]):
            continue

        cell_order = str(row["cell_order"]).split()

        # Filter to valid cells present in the JSON
        valid_order = [c for c in cell_order if c in cell_types]

        # Identify code cells
        code_cells = [c for c in valid_order if cell_types[c] == "code"]

        if not code_cells:
            continue

        # Strategy: Iterate backwards to easily find the nearest subsequent code cell.
        next_code_cell_id = None
        # Fallback for markdowns at the end: pair with the last code cell in the notebook
        last_code_cell_id = code_cells[-1]

        for cell_id in reversed(valid_order):
            c_type = cell_types[cell_id]

            if c_type == "code":
                next_code_cell_id = cell_id
            elif c_type == "markdown":
                target_code_id = (
                    next_code_cell_id
                    if next_code_cell_id is not None
                    else last_code_cell_id
                )

                md_text = sources.get(cell_id, "")
                code_text = sources.get(target_code_id, "")

                if md_text.strip() == "":
                    continue

                pairs_data.append({"markdown_text": md_text, "code_text": code_text})

    pairs_df = pd.DataFrame(pairs_data)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    pairs_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(pairs_df)} pairs to {cache_path}")

    return pairs_df


def prepare_ranking_data(
    metadata_path, mode="train", debug=False, load_cached_data=True
):
    """
    Prepares the labeled dataset for the regression task (Stage 2).
    Extracts markdown cells and their target rank.
    Target Rank = Number of code cells appearing before the markdown cell.

    Args:
        metadata_path (str): Path to the metadata CSV.
        mode (str): 'train' or 'val', used for cache naming.
        debug (bool): If True, process only a small subset.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'cell_id', 'markdown_text', 'rank', 'n_code'].
    """
    cache_filename = f"ranks_{mode}{'_debug' if debug else ''}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached ranking data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Preparing ranking data for {mode} (debug={debug})...")

    loader = NotebookLoader(metadata_path)
    df = loader.df

    if debug:
        df = df.head(100)

    ranking_data = []

    for _, row in df.iterrows():
        try:
            data = read_notebook(row["file_path"])
        except Exception:
            continue

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        if "cell_order" not in row or pd.isna(row["cell_order"]):
            continue

        cell_order = str(row["cell_order"]).split()
        valid_order = [c for c in cell_order if c in cell_types]

        code_cells = [c for c in valid_order if cell_types[c] == "code"]
        n_code = len(code_cells)

        current_rank = 0

        for cell_id in valid_order:
            c_type = cell_types[cell_id]

            if c_type == "code":
                current_rank += 1
            elif c_type == "markdown":
                ranking_data.append(
                    {
                        "id": row["id"],
                        "cell_id": cell_id,
                        "markdown_text": sources.get(cell_id, ""),
                        "rank": current_rank,
                        "n_code": n_code,
                    }
                )

    ranking_df = pd.DataFrame(ranking_data)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    ranking_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(ranking_df)} ranking samples to {cache_path}")

    return ranking_df


def prepare_test_data(metadata_path, debug=False, load_cached_data=True):
    """
    Prepares the inference dataset.
    Extracts markdown cells from test notebooks.
    Assumes code cells in test JSONs are provided in their correct relative order
    (based on standard competition format where keys are iterated).

    Args:
        metadata_path (str): Path to the test metadata CSV.
        debug (bool): If True, process only a small subset.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'cell_id', 'markdown_text', 'n_code'].
    """
    cache_filename = f"test_data{'_debug' if debug else ''}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Preparing test data (debug={debug})...")

    loader = NotebookLoader(metadata_path)
    df = loader.df

    if debug:
        df = df.head(100)

    test_data = []

    for _, row in df.iterrows():
        try:
            data = read_notebook(row["file_path"])
        except Exception:
            continue

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        # In test notebooks, we assume the JSON key iteration order preserves
        # the structure where Code cells are ordered correctly relative to each other.
        # We iterate the keys to count code cells.

        n_code = 0
        for cell_id, c_type in cell_types.items():
            if c_type == "code":
                n_code += 1

        # Extract markdown cells for prediction
        for cell_id, c_type in cell_types.items():
            if c_type == "markdown":
                test_data.append(
                    {
                        "id": row["id"],
                        "cell_id": cell_id,
                        "markdown_text": sources.get(cell_id, ""),
                        "n_code": n_code,
                    }
                )

    test_df = pd.DataFrame(test_data)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    test_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(test_df)} test samples to {cache_path}")

    return test_df
