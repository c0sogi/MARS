import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List, Dict, Any, Optional

from library.config import Config
from library.utils import read_notebook, get_ranks, load_or_process_data


def process_notebook_row(row: pd.Series, input_dir: str) -> List[Dict[str, Any]]:
    """
    Worker function to process a single notebook row from the metadata DataFrame.
    Reads the JSON file and converts it into a list of cell dictionaries.
    """
    nb_id = row["id"]
    # Filepath is relative in metadata (e.g., "train/xxxxx.json")
    filepath = os.path.join(input_dir, row["filepath"])

    try:
        nb_json = read_notebook(filepath)
    except Exception:
        # Return empty list if file read fails
        return []

    cell_types = nb_json.get("cell_type", {})
    sources = nb_json.get("source", {})

    # Determine ground truth ranks if 'cell_order' is available (Train/Val sets)
    rank_map = {}
    if "cell_order" in row and pd.notna(row["cell_order"]):
        cell_order = str(row["cell_order"]).split()
        rank_map = get_ranks(cell_order)

    # Ancestor ID is used for grouping; default to nb_id if missing (e.g. test set)
    ancestor_id = (
        row["ancestor_id"]
        if "ancestor_id" in row and pd.notna(row["ancestor_id"])
        else nb_id
    )

    cells_data = []

    # Iterate over all cells found in the JSON
    # Note: For training data, JSON keys are unordered or code-then-shuffled-markdown.
    # We rely on rank_map to establish ground truth order.
    for cell_id, c_type in cell_types.items():
        source = sources.get(cell_id, "")

        # Retrieve rank from map; default to NaN for test set or unlisted cells
        rank = rank_map.get(cell_id, np.nan)

        cells_data.append(
            {
                "id": nb_id,
                "cell_id": cell_id,
                "cell_type": c_type,
                "source": source,
                "rank": rank,
                "ancestor_id": ancestor_id,
            }
        )

    return cells_data


class DataManager:
    """
    Manages data loading, processing, and caching for the notebook sorting task.
    """

    def __init__(self):
        self.input_dir = Config.INPUT_DIR
        self.working_dir = Config.WORKING_DIR

    def load_metadata(self):
        """
        Loads the train, validation, and test metadata CSVs provided in the environment.
        """
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        return df_train, df_val, df_test

    def _generate_dataset(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to generate a cell-level DataFrame from notebook metadata.
        Uses parallel processing to read and parse JSON files.
        """
        # Execute parallel processing
        results = Parallel(n_jobs=Config.NUM_WORKERS)(
            delayed(process_notebook_row)(row, self.input_dir)
            for _, row in meta_df.iterrows()
        )

        # Flatten the list of lists
        flat_results = [item for sublist in results for item in sublist]

        # Create DataFrame
        df = pd.DataFrame(flat_results)

        # Optimize types
        if not df.empty:
            df["cell_type"] = df["cell_type"].astype("category")

        return df

    def get_train_data(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Returns the processed training data (cell-level).
        """
        df_train_meta, _, _ = self.load_metadata()
        cache_path = os.path.join(self.working_dir, "train_dataframe.parquet")

        return load_or_process_data(
            cache_path=cache_path,
            process_fn=self._generate_dataset,
            load_cached_data=load_cached_data,
            meta_df=df_train_meta,
        )

    def get_val_data(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Returns the processed validation data (cell-level).
        """
        _, df_val_meta, _ = self.load_metadata()
        cache_path = os.path.join(self.working_dir, "val_dataframe.parquet")

        return load_or_process_data(
            cache_path=cache_path,
            process_fn=self._generate_dataset,
            load_cached_data=load_cached_data,
            meta_df=df_val_meta,
        )

    def get_test_data(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Returns the processed test data (cell-level).
        """
        _, _, df_test_meta = self.load_metadata()
        cache_path = os.path.join(self.working_dir, "test_dataframe.parquet")

        return load_or_process_data(
            cache_path=cache_path,
            process_fn=self._generate_dataset,
            load_cached_data=load_cached_data,
            meta_df=df_test_meta,
        )
