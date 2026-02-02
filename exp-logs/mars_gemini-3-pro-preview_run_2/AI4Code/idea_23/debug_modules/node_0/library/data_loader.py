import os
import json
import pandas as pd
import numpy as np
from library.config import Config


class NotebookLoader:
    """
    Handles loading and processing of notebook data for the ranking task.
    Implements caching to Parquet to speed up subsequent runs.
    """

    def __init__(self):
        """
        Initialize the loader with configuration.
        """
        self.config = Config
        # Ensure working directories exist
        self.config.setup()

    def load_train_data(self, load_cached_data=True, sample_fraction=None):
        """
        Loads the training dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from cached parquet file.
            sample_fraction (float, optional): If provided, subsamples the data (0.0 to 1.0).

        Returns:
            pd.DataFrame: DataFrame containing training cells and targets.
        """
        cache_path = os.path.join(self.config.WORKING_DIR, "train_cells.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading train data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print("Processing train data from raw files...")
            df = self._process_notebooks(self.config.TRAIN_METADATA_PATH, mode="train")
            print(f"Saving train data to cache: {cache_path}")
            df.to_parquet(cache_path, index=False)

        if sample_fraction is not None:
            # Subsample based on notebook_id to keep notebooks intact
            unique_ids = df["id"].unique()
            sample_n = int(len(unique_ids) * sample_fraction)
            sampled_ids = np.random.choice(unique_ids, sample_n, replace=False)
            df = df[df["id"].isin(sampled_ids)].reset_index(drop=True)
            print(f"Subsampled train data to {len(df)} cells ({sample_fraction*100}%)")

        return df

    def load_val_data(self, load_cached_data=True):
        """
        Loads the validation dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from cached parquet file.

        Returns:
            pd.DataFrame: DataFrame containing validation cells and targets.
        """
        cache_path = os.path.join(self.config.WORKING_DIR, "val_cells.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading val data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print("Processing val data from raw files...")
            df = self._process_notebooks(
                self.config.VAL_METADATA_PATH, mode="train"
            )  # 'train' mode because val has ground truth
            print(f"Saving val data to cache: {cache_path}")
            df.to_parquet(cache_path, index=False)

        return df

    def load_test_data(self, load_cached_data=True):
        """
        Loads the test dataset.

        Args:
            load_cached_data (bool): If True, attempts to load from cached parquet file.

        Returns:
            pd.DataFrame: DataFrame containing test cells.
        """
        cache_path = os.path.join(self.config.WORKING_DIR, "test_cells.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading test data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print("Processing test data from raw files...")
            df = self._process_notebooks(self.config.TEST_METADATA_PATH, mode="test")
            print(f"Saving test data to cache: {cache_path}")
            df.to_parquet(cache_path, index=False)

        return df

    def _process_notebooks(self, metadata_path, mode="train"):
        """
        Internal method to parse JSON notebooks based on metadata.

        Args:
            metadata_path (str): Path to the metadata CSV.
            mode (str): 'train' (expects ground truth) or 'test' (no ground truth).

        Returns:
            pd.DataFrame: Processed cell data.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Pre-allocate lists for data collection
        all_data = []

        # Iterate through metadata
        # Using simple loop as progress bars are discouraged in final output
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            rel_path = row["filepath"]
            full_path = os.path.join(self.config.INPUT_DIR, rel_path)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    nb_json = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {full_path}: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine processing logic based on mode
            if mode == "train":
                # In train mode, we use the ground truth 'cell_order' string
                if pd.isna(row.get("cell_order")):
                    continue

                correct_order = row["cell_order"].split()
                total_cells = len(correct_order)

                for rank, cell_id in enumerate(correct_order):
                    c_type = cell_types.get(cell_id, "unknown")
                    c_source = sources.get(cell_id, "")

                    # Calculate normalized rank (0.0 to 1.0)
                    # If only 1 cell, rank is 0.0
                    norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

                    all_data.append(
                        {
                            "id": nb_id,
                            "cell_id": cell_id,
                            "cell_type": c_type,
                            "source": c_source,
                            "rank": float(norm_rank),
                            "ancestor_id": row.get("ancestor_id", nb_id),
                        }
                    )

            else:
                # In test mode, we don't have 'cell_order'.
                # However, code cells are in correct relative order in the JSON keys.
                # Markdown cells are shuffled/unordered relative to code.
                # We extract all cells. Rank is set to -1.0 (unknown).
                # We rely on the order of keys in the JSON for the code skeleton.

                # Iterate over keys in the JSON (Python 3.7+ preserves insertion order)
                # We assume the JSON file respects the code cell order.
                for cell_id in cell_types.keys():
                    c_type = cell_types.get(cell_id, "unknown")
                    c_source = sources.get(cell_id, "")

                    all_data.append(
                        {
                            "id": nb_id,
                            "cell_id": cell_id,
                            "cell_type": c_type,
                            "source": c_source,
                            "rank": -1.0,  # Placeholder for test
                            "ancestor_id": nb_id,  # No ancestor info in test
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Optimize types
        df["cell_type"] = df["cell_type"].astype("category")
        df["rank"] = df["rank"].astype("float32")

        return df
