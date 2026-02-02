import os
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library.config import Config


class NotebookLoader:
    """
    Handles loading and parsing of notebook data from JSON files.
    Implements caching and parallel processing for efficiency.
    """

    @staticmethod
    def _read_json(filepath):
        """
        Reads a JSON file from the given path.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _process_single_notebook(row, input_dir):
        """
        Processes a single notebook metadata row.
        Extracts cells, determines types, and calculates ranks if ground truth is available.

        Args:
            row (pd.Series): A row from the metadata DataFrame.
            input_dir (str): Base directory for input files.

        Returns:
            list: A list of dictionaries, each representing a cell.
        """
        notebook_id = row["id"]
        rel_path = row["filepath"]
        full_path = os.path.join(input_dir, rel_path)

        # Load JSON content
        data = NotebookLoader._read_json(full_path)
        if data is None:
            return []

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        # Determine if we have ground truth order
        # 'cell_order' is a space-delimited string in the metadata
        ground_truth_order = None
        if "cell_order" in row and pd.notna(row["cell_order"]):
            ground_truth_order = row["cell_order"].split()

        ancestor_id = row.get("ancestor_id", notebook_id)

        cells_data = []

        if ground_truth_order:
            # Training/Validation mode: Use the correct order to assign ranks
            total_cells = len(ground_truth_order)
            for rank, cell_id in enumerate(ground_truth_order):
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                # Calculate normalized rank (0.0 to 1.0)
                # If there is only 1 cell, norm_rank is 0.0
                norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

                cells_data.append(
                    {
                        "notebook_id": notebook_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "source": c_source,
                        "rank": rank,
                        "norm_rank": norm_rank,
                        "ancestor_id": ancestor_id,
                    }
                )
        else:
            # Test mode: No order provided, load all cells found in JSON
            # Order in JSON keys is not guaranteed to be meaningful, but we load them all.
            # Ranks are set to NaN.
            all_cell_ids = list(cell_types.keys())
            for cell_id in all_cell_ids:
                c_type = cell_types[cell_id]
                c_source = sources.get(cell_id, "")

                cells_data.append(
                    {
                        "notebook_id": notebook_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "source": c_source,
                        "rank": np.nan,
                        "norm_rank": np.nan,
                        "ancestor_id": ancestor_id,  # Likely just notebook_id for test
                    }
                )

        return cells_data

    @classmethod
    def load_dataset(
        cls, metadata_path, cache_name, load_cached_data=True, debug=False
    ):
        """
        Main entry point to load a dataset (train, val, or test).

        Args:
            metadata_path (str): Path to the metadata CSV file.
            cache_name (str): Name of the cache file (e.g., 'train_cells').
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): If True, processes a small subset of the data.

        Returns:
            pd.DataFrame: DataFrame containing processed cell data.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        cache_file = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            try:
                df = pd.read_parquet(cache_file)
                # If debug is on, we still need to slice the cached data to match expected size
                if debug:
                    # We filter by notebook_id to ensure we get complete notebooks
                    unique_ids = df["notebook_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
                    df = df[df["notebook_id"].isin(unique_ids)].reset_index(drop=True)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from scratch
        print(f"Processing data from {metadata_path}...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Handle Debugging
        if debug:
            df_meta = df_meta.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
            print(f"Debug mode: Processing first {len(df_meta)} notebooks.")

        # Parallel processing of notebooks
        # We use joblib to parallelize the file I/O and parsing
        results = Parallel(n_jobs=Config.NUM_WORKERS, backend="loky")(
            delayed(cls._process_single_notebook)(row, Config.INPUT_DIR)
            for _, row in df_meta.iterrows()
        )

        # Flatten the list of lists
        flat_results = [item for sublist in results for item in sublist]

        df = pd.DataFrame(flat_results)

        # Optimize types
        df["cell_type"] = df["cell_type"].astype("category")
        df["notebook_id"] = df["notebook_id"].astype("category")
        # ancestor_id might be mixed types in raw data, force string then category
        df["ancestor_id"] = df["ancestor_id"].astype(str).astype("category")

        # 3. Save to cache
        print(f"Saving processed data to {cache_file}...")
        df.to_parquet(cache_file, index=False)

        return df
