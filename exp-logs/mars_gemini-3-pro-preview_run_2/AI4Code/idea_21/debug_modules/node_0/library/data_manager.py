import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_json_file


class DataManager:
    """
    Manages data loading, parsing, and caching for the Notebook Cell Ordering task.
    Flattens hierarchical notebook JSONs into cell-level DataFrames.
    """

    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.WORKING_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, split_name):
        """
        Returns the path to the cached parquet file for a given split.
        """
        return os.path.join(self.cache_dir, f"{split_name}_processed.parquet")

    def load_data(self, split="train", load_cached_data=True):
        """
        Loads the dataset for the specified split (train, val, test).

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from parquet cache first.

        Returns:
            pd.DataFrame: A DataFrame containing cell-level data.
                          Columns: [id, cell_id, cell_type, source, rank, pct_rank, ancestor_id, parent_id]
        """
        cache_path = self._get_cache_path(split)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Load Metadata
        if split == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = self.config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = self.config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        print(f"Processing {split} data from source...")
        df_meta = pd.read_csv(meta_path)

        # 3. Process Notebooks
        cell_data = self._process_notebooks(df_meta, is_train=(split != "test"))

        # 4. Create DataFrame
        df = pd.DataFrame(cell_data)

        # Optimize types
        df["cell_type"] = df["cell_type"].astype("category")
        if "rank" in df.columns:
            df["rank"] = df["rank"].fillna(-1).astype(int)
            df["pct_rank"] = df["pct_rank"].astype(float)

        # 5. Save to Cache
        print(f"Saving {split} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

        return df

    def _process_notebooks(self, df_meta, is_train=True):
        """
        Iterates through the metadata DataFrame, parses JSON files, and extracts cell data.

        Args:
            df_meta (pd.DataFrame): Metadata containing 'id', 'filepath', and optionally 'cell_order'.
            is_train (bool): Whether to compute ground truth ranks.

        Returns:
            list: A list of dictionaries, each representing a cell.
        """
        data = []

        # Pre-fetch columns to avoid overhead in loop
        ids = df_meta["id"].values
        filepaths = df_meta["filepath"].values

        # Optional columns
        orders = (
            df_meta["cell_order"].values if "cell_order" in df_meta.columns else None
        )
        ancestors = (
            df_meta["ancestor_id"].values if "ancestor_id" in df_meta.columns else None
        )
        parents = (
            df_meta["parent_id"].values if "parent_id" in df_meta.columns else None
        )

        for idx in range(len(df_meta)):
            nb_id = ids[idx]
            rel_path = filepaths[idx]
            full_path = os.path.join(self.config.INPUT_DIR, rel_path)

            # Load JSON
            nb_json = read_json_file(full_path)
            if not nb_json:
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine processing order and ranks
            if is_train and orders is not None:
                order_str = orders[idx]
                if not isinstance(order_str, str):
                    continue

                cell_order = order_str.split()
                total_cells = len(cell_order)

                # Create rank lookup
                rank_map = {cid: i for i, cid in enumerate(cell_order)}

                # Iterate over the ordered list to ensure we capture the ground truth structure
                cells_to_process = cell_order
            else:
                # For test set, we just take all keys available in the source/cell_type
                # Order in JSON keys is not guaranteed to be correct for Markdown,
                # but Code cells are usually ordered.
                # We process all keys found.
                cells_to_process = list(sources.keys())
                total_cells = len(cells_to_process)
                rank_map = {}

            # Ancestor info
            anc_id = ancestors[idx] if ancestors is not None else nb_id
            par_id = parents[idx] if parents is not None else np.nan

            for cell_id in cells_to_process:
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                row = {
                    "id": nb_id,
                    "cell_id": cell_id,
                    "cell_type": c_type,
                    "source": c_source,
                    "ancestor_id": anc_id,
                    "parent_id": par_id,
                }

                if is_train:
                    rank = rank_map.get(cell_id, -1)
                    row["rank"] = rank
                    # Normalized rank: 0.0 to 1.0
                    # Avoid division by zero
                    row["pct_rank"] = (
                        rank / (total_cells - 1) if total_cells > 1 else 0.0
                    )

                data.append(row)

        return data
