import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook_json


class DataManager:
    """
    Handles loading and preprocessing of notebook data from JSON files and metadata CSVs.
    Manages caching of processed DataFrames to Parquet files.
    """

    def __init__(self):
        self.config = Config
        # Define cache filenames
        self.cache_files = {
            "train": os.path.join(self.config.CACHE_DIR, "train_dataframe.parquet"),
            "val": os.path.join(self.config.CACHE_DIR, "val_dataframe.parquet"),
            "test": os.path.join(self.config.CACHE_DIR, "test_dataframe.parquet"),
        }
        # Define metadata paths
        self.metadata_paths = {
            "train": self.config.TRAIN_METADATA_PATH,
            "val": self.config.VAL_METADATA_PATH,
            "test": self.config.TEST_METADATA_PATH,
        }

    def load_data(self, split="train", load_cached_data=True):
        """
        Loads notebook data for the specified split.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: DataFrame containing processed cell data.
        """
        if split not in self.cache_files:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        cache_path = self.cache_files[split]

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

        # 2. Process from raw files
        print(f"Processing raw {split} data...")
        metadata_path = self.metadata_paths[split]
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Handle Debugging
        if self.config.DEBUG:
            print(
                f"DEBUG mode enabled. Sampling {self.config.DEBUG_SAMPLES} notebooks."
            )
            if len(df_meta) > self.config.DEBUG_SAMPLES:
                df_meta = df_meta.sample(
                    n=self.config.DEBUG_SAMPLES, random_state=self.config.RANDOM_STATE
                ).copy()

        # Process notebooks
        processed_data = self._process_notebooks(df_meta, split)
        df = pd.DataFrame(processed_data)

        # Optimize types
        df["rank"] = df["rank"].astype(np.float32)
        df["cell_type"] = df["cell_type"].astype("category")

        # 3. Save to cache
        print(f"Saving {split} data to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

        return df

    def _process_notebooks(self, df_meta, split):
        """
        Iterates through metadata, reads JSONs, and extracts cell information.
        """
        data_list = []

        # Pre-compute common paths to avoid repeated joins in loop
        input_dir = self.config.INPUT_DIR

        # Iterate over metadata
        # Note: Not using tqdm as per instructions to avoid progress bars
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            rel_path = row["filepath"]
            full_path = os.path.join(input_dir, rel_path)

            # Ancestor ID is relevant for grouping in train/val, not present in test
            ancestor_id = row["ancestor_id"] if "ancestor_id" in row else nb_id

            # Load JSON
            nb_json = read_notebook_json(full_path)
            if nb_json is None:
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine Cell Order and Ranks
            cell_order_str = None
            if split == "test":
                # For test, we don't have ground truth order.
                # We extract all cells present in the JSON keys.
                # Note: In the provided dataset, 'cell_type' keys usually contain all cell IDs.
                cell_ids = list(cell_types.keys())
                rank_map = {}  # No ranks for test
            else:
                # For train/val, we have ground truth order
                if pd.isna(row["cell_order"]):
                    continue
                cell_order_str = row["cell_order"]
                cell_order = cell_order_str.split()
                num_cells = len(cell_order)

                # Calculate normalized rank: 0.0 to 1.0
                if num_cells > 1:
                    rank_map = {
                        cid: i / (num_cells - 1) for i, cid in enumerate(cell_order)
                    }
                else:
                    rank_map = {cid: 0.0 for cid in cell_order}

                cell_ids = cell_order

            # Extract data for each cell
            for cell_id in cell_ids:
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                # Ensure source is a string
                if not isinstance(c_source, str):
                    c_source = str(c_source)

                # Get rank (NaN for test)
                c_rank = rank_map.get(cell_id, np.nan)

                data_list.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "source": c_source,
                        "rank": c_rank,
                        "ancestor_id": ancestor_id,
                        "cell_order": cell_order_str,
                    }
                )

        return data_list
