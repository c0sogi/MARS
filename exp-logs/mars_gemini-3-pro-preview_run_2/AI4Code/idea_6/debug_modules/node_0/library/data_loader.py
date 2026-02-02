import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook


def get_cell_ranks(cell_order):
    """
    Computes the normalized rank for each cell in the notebook.

    Args:
        cell_order (list): List of cell IDs in the correct order.

    Returns:
        dict: Mapping from cell_id to normalized rank (0.0 to 1.0).
    """
    n_cells = len(cell_order)
    if n_cells <= 1:
        return {cell_id: 0.0 for cell_id in cell_order}

    ranks = {}
    for rank, cell_id in enumerate(cell_order):
        ranks[cell_id] = rank / (n_cells - 1)
    return ranks


class NotebookDataLoader:
    """
    Handles loading, processing, and caching of notebook data for the ranking task.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.debug_size = Config.DEBUG_SAMPLE_SIZE

    def load_data(self, split="train", load_cached_data=True):
        """
        Loads data for a specific split (train, val, test).

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (df_markdown, df_notebooks)
                - df_markdown: DataFrame containing markdown cells and targets.
                - df_notebooks: DataFrame containing notebook-level context and structure.
        """
        # Define cache paths
        cache_md_path = os.path.join(Config.WORKING_DIR, f"{split}_markdown.parquet")
        cache_nb_path = os.path.join(Config.WORKING_DIR, f"{split}_notebooks.parquet")

        # Try loading from cache
        if (
            load_cached_data
            and os.path.exists(cache_md_path)
            and os.path.exists(cache_nb_path)
        ):
            print(f"Loading {split} data from cache...")
            df_md = pd.read_parquet(cache_md_path)
            df_nb = pd.read_parquet(cache_nb_path)
            return df_md, df_nb

        print(f"Processing {split} data from scratch...")

        # Load metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df_meta = pd.read_csv(meta_path)

        # Apply debug sampling
        if self.debug:
            df_meta = df_meta.head(self.debug_size)

        # Process notebooks
        md_records = []
        nb_records = []

        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = row["filepath"]

            # Load notebook JSON
            try:
                nb_json = read_notebook(filepath)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine structure based on split
            if split in ["train", "val"]:
                # Use ground truth order
                cell_order_str = row["cell_order"]
                if pd.isna(cell_order_str):
                    continue
                correct_order = cell_order_str.split()

                # Calculate ranks
                ranks = get_cell_ranks(correct_order)

                code_cells = []

                for cell_id in correct_order:
                    ctype = cell_types.get(cell_id, "unknown")
                    source = sources.get(cell_id, "")

                    if ctype == "markdown":
                        md_records.append(
                            {
                                "id": nb_id,
                                "cell_id": cell_id,
                                "source": source,
                                "rank": ranks.get(cell_id, 0.0),
                            }
                        )
                    elif ctype == "code":
                        code_cells.append(source)

                # Notebook level info
                nb_records.append(
                    {
                        "id": nb_id,
                        "code_context": " ".join(code_cells),
                        "total_cells": len(correct_order),
                        "code_cell_ids": "",  # Not needed for train/val
                    }
                )

            else:  # test
                # For test, we rely on the file structure:
                # Code cells are first and ordered. Markdown cells are shuffled after.
                # We iterate the keys of the dict (insertion order preserved in Python 3.7+)
                all_cells = list(cell_types.keys())

                code_cell_ids = []
                code_sources = []

                # First pass: Identify code cells (which are ordered correctly)
                for cell_id in all_cells:
                    if cell_types.get(cell_id) == "code":
                        code_cell_ids.append(cell_id)
                        code_sources.append(sources.get(cell_id, ""))

                # Second pass: Identify markdown cells (which are shuffled)
                for cell_id in all_cells:
                    if cell_types.get(cell_id) == "markdown":
                        md_records.append(
                            {
                                "id": nb_id,
                                "cell_id": cell_id,
                                "source": sources.get(cell_id, ""),
                                "rank": np.nan,  # No target for test
                            }
                        )

                nb_records.append(
                    {
                        "id": nb_id,
                        "code_context": " ".join(code_sources),
                        "total_cells": len(all_cells),
                        "code_cell_ids": " ".join(code_cell_ids),
                    }
                )

        # Create DataFrames
        df_markdown = pd.DataFrame(md_records)
        df_notebooks = pd.DataFrame(nb_records)

        # Ensure schema consistency
        if "rank" not in df_markdown.columns:
            df_markdown["rank"] = np.nan

        # Cache results
        print(f"Saving {split} data to cache...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df_markdown.to_parquet(cache_md_path, index=False)
        df_notebooks.to_parquet(cache_nb_path, index=False)

        return df_markdown, df_notebooks
