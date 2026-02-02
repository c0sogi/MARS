import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook


class NotebookProcessor:
    """
    Handles the loading, parsing, and processing of notebook data from JSON files
    into flattened DataFrames suitable for feature engineering and modeling.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def load_train_data(self, load_cached_data=True):
        """
        Loads and processes the training dataset.
        """
        cache_path = os.path.join(self.working_dir, "train_processed.parquet")
        return self._process_split(
            metadata_path=Config.TRAIN_METADATA_PATH,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
            is_test=False,
        )

    def load_val_data(self, load_cached_data=True):
        """
        Loads and processes the validation dataset.
        """
        cache_path = os.path.join(self.working_dir, "val_processed.parquet")
        return self._process_split(
            metadata_path=Config.VAL_METADATA_PATH,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
            is_test=False,
        )

    def load_test_data(self, load_cached_data=True):
        """
        Loads and processes the test dataset.
        """
        cache_path = os.path.join(self.working_dir, "test_processed.parquet")
        return self._process_split(
            metadata_path=Config.TEST_METADATA_PATH,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
            is_test=True,
        )

    def _process_split(self, metadata_path, cache_path, load_cached_data, is_test):
        """
        Internal method to process a specific data split (train/val/test).
        Implements the caching logic.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from scratch
        print(f"Processing data from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        # Pre-allocate lists for speed
        all_cells = []

        # Iterate over metadata
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = row["filepath"]

            # Load JSON content
            try:
                nb_json = read_notebook(filepath)
            except Exception:
                # Skip corrupt files if any
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine cell order
            if is_test:
                # For test, we just take all keys present in cell_type
                # The order in the JSON is not guaranteed to be correct,
                # but we need to extract them all.
                cell_order = list(cell_types.keys())
                correct_ranks = {}  # No ground truth
            else:
                # For train/val, use the provided ground truth order
                if pd.isna(row["cell_order"]):
                    continue
                cell_order = row["cell_order"].split()
                # Create a map of cell_id -> rank (integer index)
                correct_ranks = {cid: i for i, cid in enumerate(cell_order)}

            total_cells = len(cell_order)

            # Process each cell
            for cell_id in cell_order:
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                # Determine rank and normalized rank
                if is_test:
                    rank = -1
                    pct_rank = -1.0
                else:
                    rank = correct_ranks.get(cell_id, -1)
                    # Normalized rank [0, 1]
                    if total_cells > 1:
                        pct_rank = rank / (total_cells - 1)
                    else:
                        pct_rank = 0.0

                # Append to list
                all_cells.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "source": c_source,
                        "rank": rank,
                        "pct_rank": pct_rank,
                        "ancestor_id": (
                            row.get("ancestor_id", nb_id) if not is_test else nb_id
                        ),
                    }
                )

        # Create DataFrame
        df_processed = pd.DataFrame(all_cells)

        # Optimize types
        df_processed["cell_type"] = df_processed["cell_type"].astype("category")

        # 3. Save to cache
        print(f"Saving processed data to {cache_path}...")
        df_processed.to_parquet(cache_path, index=False)

        return df_processed
