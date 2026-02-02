import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import load_processed_cells


def normalize_ranks(ranks):
    """
    Converts a list or array of integer positions to the [0, 1] range.

    Args:
        ranks (list or np.array): Integer ranks (0 to N-1).

    Returns:
        np.array: Normalized ranks.
    """
    ranks = np.array(ranks)
    n = len(ranks)
    if n <= 1:
        return np.zeros_like(ranks, dtype=float)
    return ranks / (n - 1)


def flatten_notebooks(df_metadata, split_name="train", load_cached_data=True):
    """
    Converts nested notebook JSONs (referenced in metadata) into a flat DataFrame.
    Each row represents a cell with its source, type, and extracted identifiers.

    Delegates to library.utils.load_processed_cells which handles:
    - Reading JSONs
    - Extracting source code and identifiers
    - Computing ground truth ranks (for train/val)
    - Caching the result to Parquet

    Args:
        df_metadata (pd.DataFrame): DataFrame containing 'id' and 'filepath'.
        split_name (str): Name of the split ('train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to use cached Parquet files.

    Returns:
        pd.DataFrame: Flattened cell-level data.
    """
    return load_processed_cells(
        df_metadata, split_name=split_name, load_cached_data=load_cached_data
    )


class NotebookLoader:
    """
    Manager class for loading notebook metadata and flattening data for specific splits.
    """

    def __init__(self):
        self.metadata_paths = {
            "train": Config.TRAIN_METADATA_PATH,
            "val": Config.VAL_METADATA_PATH,
            "test": Config.TEST_METADATA_PATH,
        }

    def load_metadata(self, split):
        """
        Loads the metadata CSV for a specific split.

        Args:
            split (str): One of 'train', 'val', 'test'.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        if split not in self.metadata_paths:
            raise ValueError(
                f"Unknown split: {split}. Must be one of {list(self.metadata_paths.keys())}"
            )

        path = self.metadata_paths[split]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        return pd.read_csv(path)

    def get_flattened_data(self, split="train", load_cached_data=True):
        """
        High-level pipeline to load metadata and return flattened cell data.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to use cached Parquet files.

        Returns:
            pd.DataFrame: Flattened data containing cell sources, types, and ranks.
        """
        df_meta = self.load_metadata(split)

        # Pass the split name to flatten_notebooks to enable correct caching
        df_flat = flatten_notebooks(
            df_meta, split_name=split, load_cached_data=load_cached_data
        )

        return df_flat
