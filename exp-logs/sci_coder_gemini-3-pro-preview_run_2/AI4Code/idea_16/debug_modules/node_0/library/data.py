import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Any
from library.config import Config
from library.utils import get_logger


class NotebookLoader:
    """
    Handles loading, parsing, and preprocessing of notebook data.
    Manages caching of processed DataFrames to disk (Parquet) to speed up subsequent runs.
    """

    def __init__(self):
        self.logger = get_logger("data_loader")
        self.input_dir = Path(Config.INPUT_DIR)
        self.metadata_dir = Path(Config.METADATA_DIR)
        self.working_dir = Path(Config.WORKING_DIR)

        # Ensure working directory exists for caching
        os.makedirs(self.working_dir, exist_ok=True)

    def load_dataset(
        self, split: str, load_cached_data: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the dataset for a specific split (train, val, test).

        Returns two DataFrames:
        1. df_markdown: Contains individual markdown cells with their content and target rank.
        2. df_notebooks: Contains notebook-level context (ordered code cells).

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from Parquet cache.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_markdown, df_notebooks)
        """
        md_cache_path = self.working_dir / f"{split}_markdown.parquet"
        nb_cache_path = self.working_dir / f"{split}_notebooks.parquet"

        # 1. Try to load from cache
        if load_cached_data and md_cache_path.exists() and nb_cache_path.exists():
            self.logger.info(f"Loading {split} data from cache...")
            try:
                df_md = pd.read_parquet(md_cache_path)
                df_nb = pd.read_parquet(nb_cache_path)
                return df_md, df_nb
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cache for {split}: {e}. Reprocessing..."
                )

        # 2. Process from scratch
        self.logger.info(f"Processing {split} data from raw files...")
        df_md, df_nb = self._process_raw_data(split)

        # 3. Save to cache
        self.logger.info(f"Caching {split} data to {self.working_dir}...")
        df_md.to_parquet(md_cache_path, index=False)
        df_nb.to_parquet(nb_cache_path, index=False)

        return df_md, df_nb

    def _read_notebook(self, filepath: Path) -> Tuple[Dict, Dict]:
        """
        Parses a single JSON notebook file.

        Args:
            filepath (Path): Path to the JSON file.

        Returns:
            Tuple[Dict, Dict]: (cell_types, sources) dictionaries.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})
        return cell_types, sources

    def _process_raw_data(self, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Internal method to iterate over metadata, parse notebooks, and construct DataFrames.
        """
        # Load Metadata
        meta_path = self.metadata_dir / f"{split}_metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Debugging: Sample subset if configured
        if Config.DEBUG:
            self.logger.info(
                f"Debug mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks."
            )
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        md_rows = []
        nb_rows = []

        # Iterate over notebooks
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            rel_path = row["filepath"]
            full_path = self.input_dir / rel_path

            try:
                cell_types, sources = self._read_notebook(full_path)
            except Exception as e:
                self.logger.warning(f"Error reading notebook {nb_id}: {e}")
                continue

            # Determine Code Sequence and Markdown Targets
            code_ids = []
            md_ids_with_rank = []  # List of (cell_id, rank)

            if split in ["train", "val"]:
                # Use ground truth order
                if pd.isna(row["cell_order"]):
                    continue

                cell_order = row["cell_order"].split()
                total_cells = len(cell_order)

                for rank_idx, cid in enumerate(cell_order):
                    ctype = cell_types.get(cid)
                    if ctype == "code":
                        code_ids.append(cid)
                    elif ctype == "markdown":
                        # Calculate normalized rank
                        # If total_cells is 1, rank is 0.0
                        norm_rank = (
                            rank_idx / (total_cells - 1) if total_cells > 1 else 0.0
                        )
                        md_ids_with_rank.append((cid, norm_rank))

            else:
                # Test set: No ground truth order provided in metadata.
                # We assume the JSON keys for code cells are in correct relative order,
                # and markdown cells are shuffled.
                # We collect all keys.
                all_keys = list(cell_types.keys())

                # Filter by type
                code_ids = [c for c in all_keys if cell_types.get(c) == "code"]

                # For test, rank is unknown (-1.0)
                md_ids_raw = [c for c in all_keys if cell_types.get(c) == "markdown"]
                md_ids_with_rank = [(c, -1.0) for c in md_ids_raw]

            # Store Notebook Context (Code Cells)
            # We store the actual source code here. Parquet handles lists of strings efficiently.
            code_sources = [sources.get(cid, "") for cid in code_ids]

            nb_rows.append(
                {
                    "notebook_id": nb_id,
                    "code_ids": code_ids,  # List[str]
                    "code_sources": code_sources,  # List[str]
                }
            )

            # Store Markdown Cells
            for cid, rank in md_ids_with_rank:
                md_rows.append(
                    {
                        "notebook_id": nb_id,
                        "cell_id": cid,
                        "source": sources.get(cid, ""),
                        "rank": rank,
                    }
                )

        # Create DataFrames
        df_md = pd.DataFrame(md_rows)
        df_nb = pd.DataFrame(nb_rows)

        # Ensure correct types
        if not df_md.empty:
            df_md["rank"] = df_md["rank"].astype(float)

        return df_md, df_nb
