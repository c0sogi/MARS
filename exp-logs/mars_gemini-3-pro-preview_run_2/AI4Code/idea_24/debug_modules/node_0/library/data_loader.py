import os
import json
import pandas as pd
import numpy as np
from library.config import INPUT_DIR, METADATA_DIR, CACHE_DIR, SEED


class NotebookLoader:
    """
    Handles the loading, parsing, and caching of notebook data.
    """

    def __init__(self):
        pass

    def load_notebooks(
        self,
        metadata_path: str,
        partition_name: str,
        load_cached_data: bool = True,
        sample_size: int = None,
    ) -> pd.DataFrame:
        """
        Loads notebook data for a specific partition.

        Args:
            metadata_path: Path to the metadata CSV file.
            partition_name: Name of the partition (e.g., 'train', 'val', 'test').
            load_cached_data: If True, attempts to load from parquet cache first.
            sample_size: Number of notebooks to sample (for debugging).

        Returns:
            pd.DataFrame: A DataFrame containing cell-level data.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"{partition_name}_cells.parquet")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {partition_name} data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)

            # Apply sampling to cached data if requested
            if sample_size is not None:
                unique_ids = df["notebook_id"].unique()
                if len(unique_ids) > sample_size:
                    rng = np.random.default_rng(SEED)
                    selected_ids = rng.choice(
                        unique_ids, size=sample_size, replace=False
                    )
                    df = df[df["notebook_id"].isin(selected_ids)].reset_index(drop=True)
            return df

        # 2. Process from Raw Files
        print(f"Processing {partition_name} data from raw files...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Apply sampling to metadata before processing
        if sample_size is not None and sample_size < len(df_meta):
            df_meta = df_meta.sample(n=sample_size, random_state=SEED).reset_index(
                drop=True
            )

        data = []
        has_orders = "cell_order" in df_meta.columns
        has_ancestors = "ancestor_id" in df_meta.columns

        # Iterate over metadata
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            rel_path = row["filepath"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            # Default ancestor to self if not present (e.g., test set)
            ancestor_id = row["ancestor_id"] if has_ancestors else nb_id

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    nb_json = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read {full_path}. Error: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            if has_orders:
                # Training/Validation: Use ground truth order
                order_str = row["cell_order"]
                if pd.isna(order_str):
                    continue
                cell_order = order_str.split()
                total_cells = len(cell_order)

                for rank, cell_id in enumerate(cell_order):
                    c_type = cell_types.get(cell_id, "unknown")
                    c_source = sources.get(cell_id, "")

                    data.append(
                        {
                            "notebook_id": nb_id,
                            "cell_id": cell_id,
                            "cell_type": c_type,
                            "source": c_source,
                            "rank": rank,
                            "ancestor_id": ancestor_id,
                            "total_cells": total_cells,
                        }
                    )
            else:
                # Test: Order is unknown, extract all cells
                # Keys in the JSON dictionaries are the cell_ids
                all_cell_ids = list(cell_types.keys())
                total_cells = len(all_cell_ids)

                for cell_id in all_cell_ids:
                    c_type = cell_types.get(cell_id, "unknown")
                    c_source = sources.get(cell_id, "")

                    data.append(
                        {
                            "notebook_id": nb_id,
                            "cell_id": cell_id,
                            "cell_type": c_type,
                            "source": c_source,
                            "rank": -1,  # Rank is unknown for test set
                            "ancestor_id": ancestor_id,
                            "total_cells": total_cells,
                        }
                    )

        df = pd.DataFrame(data)

        # 3. Save to Cache
        print(f"Saving {partition_name} data to cache: {cache_path}")
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

        return df

    def get_partitioned_data(self, load_cached_data: bool = True, debug: bool = False):
        """
        Retrieves train, validation, and test datasets.

        Args:
            load_cached_data: Whether to use cached parquet files.
            debug: If True, loads a small subset of data for debugging.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (df_train, df_val, df_test)
        """
        sample_size = 100 if debug else None

        train_meta = os.path.join(METADATA_DIR, "train_metadata.csv")
        val_meta = os.path.join(METADATA_DIR, "val_metadata.csv")
        test_meta = os.path.join(METADATA_DIR, "test_metadata.csv")

        df_train = self.load_notebooks(
            train_meta, "train", load_cached_data, sample_size
        )
        df_val = self.load_notebooks(val_meta, "val", load_cached_data, sample_size)
        df_test = self.load_notebooks(test_meta, "test", load_cached_data, sample_size)

        return df_train, df_val, df_test
