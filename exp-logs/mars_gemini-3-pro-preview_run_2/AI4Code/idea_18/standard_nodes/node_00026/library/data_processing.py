import os
import re
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library.config import Config
from library.utils import read_notebook


class NotebookProcessor:
    """
    Handles data loading, cleaning, symbolic extraction, and dataset creation.
    Implements caching mechanisms using Parquet files to optimize runtime.
    """

    def __init__(self):
        # Compile regex for symbolic extraction once
        self.symbol_pattern = re.compile(Config.SYMBOLIC_TOKEN_PATTERN)

    def preprocess_text(self, text):
        """
        Basic text preprocessing.
        Ensures input is a string. Explicitly avoids accent stripping as per configuration.
        """
        if text is None:
            return ""
        return str(text)

    def extract_symbols(self, text):
        """
        Extracts variable and function names using the configured regex.
        Returns a space-separated string of unique symbols.
        """
        if not text:
            return ""

        # Find all matches using the regex pattern defined in Config
        matches = self.symbol_pattern.findall(text)

        # Return unique symbols as a space-separated string
        # Sorting ensures deterministic output
        unique_symbols = sorted(list(set(matches)))
        return " ".join(unique_symbols)

    def _process_row(self, row, split_type):
        """
        Helper method to process a single notebook row from metadata.
        Reads the JSON file and extracts cell-level features.

        Args:
            row: Series containing 'id', 'filepath', and optionally 'cell_order'.
            split_type: 'train', 'val', or 'test'.
        Returns:
            List of dictionaries representing cells.
        """
        nb_id = row["id"]
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Load notebook content
        nb_content = read_notebook(filepath)
        if not nb_content:
            return []

        cell_types = nb_content.get("cell_type", {})
        sources = nb_content.get("source", {})

        cells_data = []

        # Check if ground truth order is available (Train/Val)
        if "cell_order" in row and pd.notna(row["cell_order"]):
            cell_order = row["cell_order"].split()
            total_cells = len(cell_order)

            for rank, cell_id in enumerate(cell_order):
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                # Preprocess text and extract symbols
                c_source_clean = self.preprocess_text(c_source)
                symbols = self.extract_symbols(c_source_clean)

                # Calculate Normalized Rank [0, 1]
                pct_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

                cells_data.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "source": c_source_clean,
                        "rank": rank,
                        "pct_rank": pct_rank,
                        "symbols": symbols,
                        "ancestor_id": row.get("ancestor_id", nb_id),
                        "parent_id": row.get("parent_id", np.nan),
                    }
                )
        else:
            # Test case: No ground truth order provided
            # We extract all cells found in the JSON.
            # Note: Python 3.7+ preserves insertion order, so code cells
            # (which are ordered in the file) will retain relative order in the keys list.
            current_keys = list(cell_types.keys())

            for i, cell_id in enumerate(current_keys):
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                c_source_clean = self.preprocess_text(c_source)
                symbols = self.extract_symbols(c_source_clean)

                cells_data.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "source": c_source_clean,
                        "rank": -1,  # Unknown
                        "pct_rank": -1.0,  # Unknown
                        "symbols": symbols,
                        "ancestor_id": np.nan,
                        "parent_id": np.nan,
                    }
                )

        return cells_data

    def load_dataset(self, split="train", load_cached_data=True):
        """
        Loads the dataset for the specified split.
        Implements caching to parquet files in Config.WORKING_DIR.
        Validates cache consistency against metadata before loading.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            pd.DataFrame: Processed dataset with one row per cell.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(Config.WORKING_DIR, f"{split}_processed.parquet")

        # Identify metadata file path
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # Load metadata first to establish expectations
        df_meta = pd.read_csv(meta_path)

        # Apply Debug Mode Sampling if active
        if Config.DEBUG:
            print(f"DEBUG Mode: Sampling 100 notebooks from {split}...")
            df_meta = df_meta.head(100)

        # 1. Try to load and VALIDATE cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Checking cache: {cache_path}")
            try:
                df_cache = pd.read_parquet(cache_path)

                # Validate Cache Consistency (Cite debug_lesson_1)
                expected_ids = set(df_meta["id"].unique())
                cached_ids = set(df_cache["id"].unique())

                if expected_ids == cached_ids:
                    print(f"Cache valid. Loading {split} data from cache.")
                    return df_cache
                else:
                    print(
                        f"Cache mismatch! Expected {len(expected_ids)} notebooks, found {len(cached_ids)} in cache."
                    )
                    print("Invalidating cache and reprocessing...")
            except Exception as e:
                print(f"Failed to read or validate cache: {e}. Reprocessing...")

        # 2. Process from scratch
        print(f"Processing {split} data from metadata...")

        # Parallel Processing
        # Use joblib to process notebooks in parallel
        results = Parallel(n_jobs=Config.NUM_WORKERS, verbose=0)(
            delayed(self._process_row)(row, split) for _, row in df_meta.iterrows()
        )

        # Flatten results list
        flat_results = [item for sublist in results for item in sublist]

        df = pd.DataFrame(flat_results)

        # Optimize data types for memory efficiency
        if not df.empty:
            df["cell_type"] = df["cell_type"].astype("category")
            df["rank"] = df["rank"].astype(int)
            df["pct_rank"] = df["pct_rank"].astype(float)

        # 3. Save to cache
        print(f"Saving {split} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

        return df
