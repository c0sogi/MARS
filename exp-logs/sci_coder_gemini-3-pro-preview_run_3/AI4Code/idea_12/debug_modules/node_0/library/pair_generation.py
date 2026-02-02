import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_utils import get_metadata, get_notebook_cells


def generate_bidirectional_pairs(load_cached_data=True):
    """
    Generates (markdown, code) pairs for contrastive fine-tuning.
    For each markdown cell, creates pairs with:
      1. The nearest preceding code cell.
      2. The nearest succeeding code cell.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame with columns ['markdown', 'code'].
    """
    # Define cache path
    cache_path = os.path.join(
        Config.Paths.CACHE_DIR, "train_pairs_bidirectional.parquet"
    )

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pairs from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating bidirectional pairs from scratch...")

    # Load training metadata
    df_train = get_metadata("train")

    # Sample notebooks based on config
    n_samples = Config.Training.NUM_NOTEBOOKS_FINE_TUNE
    if n_samples is not None and n_samples < len(df_train):
        df_train = df_train.sample(n=n_samples, random_state=Config.SEED).reset_index(
            drop=True
        )

    pairs_data = []

    # Iterate through notebooks
    # Note: We avoid tqdm/progress bars as per instructions
    for _, row in df_train.iterrows():
        notebook_id = row["id"]
        cell_order_str = row["cell_order"]
        rel_path = row["file_path"]

        # Get cell content
        try:
            notebook_data = get_notebook_cells(notebook_id, rel_path)
        except Exception:
            # Skip if file read fails
            continue

        # Create lookup for cell content
        # Combine code and markdown into a single lookup dict
        cells_dict = {}
        for cell in notebook_data["code_cells"]:
            cells_dict[cell["id"]] = {"source": cell["source"], "type": "code"}
        for cell in notebook_data["markdown_cells"]:
            cells_dict[cell["id"]] = {"source": cell["source"], "type": "markdown"}

        # Parse ground truth order
        ordered_ids = cell_order_str.split()

        # Filter ordered_ids to only those present in the JSON (handling potential inconsistencies)
        ordered_ids = [cid for cid in ordered_ids if cid in cells_dict]

        # Identify indices of code cells in the ordered sequence
        code_indices = [
            i for i, cid in enumerate(ordered_ids) if cells_dict[cid]["type"] == "code"
        ]

        if not code_indices:
            continue

        # Iterate through the sequence to find markdown cells and their neighbors
        for i, cid in enumerate(ordered_ids):
            if cells_dict[cid]["type"] == "markdown":
                md_text = cells_dict[cid]["source"]

                # Skip empty markdown cells
                if not md_text.strip():
                    continue

                # Find nearest preceding code cell
                # We want max(idx) in code_indices where idx < i
                prev_candidates = [idx for idx in code_indices if idx < i]
                if prev_candidates:
                    prev_idx = prev_candidates[
                        -1
                    ]  # The last one is the closest preceding
                    prev_code_id = ordered_ids[prev_idx]
                    prev_code_text = cells_dict[prev_code_id]["source"]
                    if prev_code_text.strip():
                        pairs_data.append({"markdown": md_text, "code": prev_code_text})

                # Find nearest succeeding code cell
                # We want min(idx) in code_indices where idx > i
                next_candidates = [idx for idx in code_indices if idx > i]
                if next_candidates:
                    next_idx = next_candidates[
                        0
                    ]  # The first one is the closest succeeding
                    next_code_id = ordered_ids[next_idx]
                    next_code_text = cells_dict[next_code_id]["source"]
                    if next_code_text.strip():
                        pairs_data.append({"markdown": md_text, "code": next_code_text})

    # Create DataFrame
    df_pairs = pd.DataFrame(pairs_data)

    # Save to cache
    print(f"Saving {len(df_pairs)} pairs to {cache_path}")
    df_pairs.to_parquet(cache_path, index=False)

    return df_pairs
