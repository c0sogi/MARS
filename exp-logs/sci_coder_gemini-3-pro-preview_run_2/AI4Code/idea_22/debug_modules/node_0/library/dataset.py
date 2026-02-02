import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook_json


class NotebookLoader:
    """
    Handles loading and processing of notebook data from JSON files based on metadata.
    Implements caching using Parquet files to speed up subsequent runs.
    """

    def __init__(self):
        self.input_dir = Config.INPUT_DIR
        self.working_dir = Config.WORKING_DIR
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def load_train_data(self, load_cached_data=True):
        """
        Loads the training dataset.
        """
        cache_path = os.path.join(self.working_dir, "train_dataframe.parquet")
        return self._process_dataset(
            metadata_path=Config.TRAIN_METADATA_PATH,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
            is_train=True,
        )

    def load_val_data(self, load_cached_data=True):
        """
        Loads the validation dataset.
        """
        cache_path = os.path.join(self.working_dir, "val_dataframe.parquet")
        return self._process_dataset(
            metadata_path=Config.VAL_METADATA_PATH,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
            is_train=True,
        )

    def load_test_data(self, load_cached_data=True):
        """
        Loads the test dataset.
        """
        cache_path = os.path.join(self.working_dir, "test_dataframe.parquet")
        return self._process_dataset(
            metadata_path=Config.TEST_METADATA_PATH,
            cache_path=cache_path,
            load_cached_data=load_cached_data,
            is_train=False,
        )

    def _process_dataset(self, metadata_path, cache_path, load_cached_data, is_train):
        """
        Internal method to process notebooks based on metadata.
        Reads JSONs, extracts cells, computes ranks (if train), and caches the result.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached dataframe from {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        print(f"Processing data from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        data_rows = []

        # 3. Iterate over notebooks
        # We iterate through the metadata dataframe to find file paths and ground truth
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            rel_path = row["filepath"]
            full_path = os.path.join(self.input_dir, rel_path)

            # Use utility to read JSON
            try:
                nb_json = read_notebook_json(full_path)
            except Exception as e:
                print(f"Warning: Failed to read {full_path}. Error: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine ancestor_id (default to nb_id if missing, e.g. in test)
            ancestor_id = row.get("ancestor_id", nb_id)

            if is_train:
                # Training/Validation Logic: Use Ground Truth Order
                cell_order_str = row["cell_order"]
                if pd.isna(cell_order_str):
                    continue

                cell_order = cell_order_str.split()
                total_cells = len(cell_order)

                # Map cell_id to its rank index
                for rank, cell_id in enumerate(cell_order):
                    ctype = cell_types.get(cell_id, "unknown")
                    src = sources.get(cell_id, "")

                    # Compute Normalized Rank [0, 1]
                    # If there is only 1 cell, pct_rank is 0.0
                    if total_cells > 1:
                        pct_rank = rank / (total_cells - 1.0)
                    else:
                        pct_rank = 0.0

                    data_rows.append(
                        {
                            "id": nb_id,
                            "cell_id": cell_id,
                            "cell_type": ctype,
                            "source": src,
                            "rank": rank,
                            "pct_rank": pct_rank,
                            "ancestor_id": ancestor_id,
                        }
                    )
            else:
                # Test Logic: No Ground Truth Order
                # Load all cells available in the source dictionary
                all_cell_ids = list(sources.keys())

                for cell_id in all_cell_ids:
                    ctype = cell_types.get(cell_id, "unknown")
                    src = sources.get(cell_id, "")

                    data_rows.append(
                        {
                            "id": nb_id,
                            "cell_id": cell_id,
                            "cell_type": ctype,
                            "source": src,
                            "rank": -1,  # Placeholder
                            "pct_rank": -1.0,  # Placeholder
                            "ancestor_id": ancestor_id,
                        }
                    )

        # 4. Create DataFrame
        df_out = pd.DataFrame(data_rows)

        # 5. Save to Cache
        print(f"Saving processed dataframe to {cache_path} with {len(df_out)} rows.")
        df_out.to_parquet(cache_path, index=False)

        return df_out
