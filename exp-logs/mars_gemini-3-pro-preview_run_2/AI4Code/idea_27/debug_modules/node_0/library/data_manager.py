import os
import pandas as pd
from library.config import Config, load_corpus, read_notebook, get_ranks


class NotebookLoader:
    def __init__(self, config=None):
        """
        Initialize the NotebookLoader.

        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config if config else Config()
        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def parse_notebook(self, filepath):
        """
        Reads a JSON notebook and returns cell data.
        Wraps the library function read_notebook.

        Args:
            filepath (str): Path to the notebook JSON file.

        Returns:
            list: List of dictionaries containing cell data.
        """
        return read_notebook(filepath)

    def compute_normalized_ranks(self, base, derived):
        """
        Computes normalized ranks for derived elements relative to base elements.
        Wraps the library function get_ranks.

        Args:
            base (list): List of anchor elements (e.g., code cells).
            derived (list): List of elements to rank (e.g., markdown cells).

        Returns:
            list: Normalized ranks [0, 1].
        """
        return get_ranks(base, derived)

    def prepare_datasets(self, load_cached_data=True):
        """
        Loads training and validation datasets based on metadata files.
        Handles caching and splitting based on ancestor groups defined in metadata.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.
                                     If False, clears cache and re-processes.

        Returns:
            tuple: (train_corpus, val_corpus) as pandas DataFrames.
        """
        # The load_corpus function in library.config caches based on the 'mode' string.
        # It saves to {WORKING_DIR}/{mode}_corpus.parquet.
        # We use mode='train' to ensure ranks (targets) are calculated for both train and val.
        cache_file = os.path.join(self.config.WORKING_DIR, "train_corpus.parquet")

        if not load_cached_data and os.path.exists(cache_file):
            os.remove(cache_file)

        # Load Metadata
        train_meta_path = os.path.join(self.config.METADATA_DIR, "train_metadata.csv")
        val_meta_path = os.path.join(self.config.METADATA_DIR, "val_metadata.csv")

        if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
            raise FileNotFoundError(
                "Metadata files not found. Please run metadata generation first."
            )

        df_train_meta = pd.read_csv(train_meta_path)
        df_val_meta = pd.read_csv(val_meta_path)

        # Combine metadata to process all at once. This allows load_corpus to create
        # a single cache file and ensures consistent processing.
        df_full_meta = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

        # Load Corpus (this handles parsing, target generation, and caching)
        full_corpus = load_corpus(self.config, df_full_meta, mode="train")

        # Split back into train and val based on IDs to respect the ancestor split
        train_ids = set(df_train_meta["id"])
        val_ids = set(df_val_meta["id"])

        train_corpus = (
            full_corpus[full_corpus["id"].isin(train_ids)].copy().reset_index(drop=True)
        )
        val_corpus = (
            full_corpus[full_corpus["id"].isin(val_ids)].copy().reset_index(drop=True)
        )

        return train_corpus, val_corpus

    def load_test_data(self, load_cached_data=True):
        """
        Loads test dataset based on metadata.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.
                                     If False, clears cache and re-processes.

        Returns:
            pd.DataFrame: Test corpus.
        """
        cache_file = os.path.join(self.config.WORKING_DIR, "test_corpus.parquet")

        if not load_cached_data and os.path.exists(cache_file):
            os.remove(cache_file)

        test_meta_path = os.path.join(self.config.METADATA_DIR, "test_metadata.csv")
        if not os.path.exists(test_meta_path):
            raise FileNotFoundError("Test metadata file not found.")

        df_test_meta = pd.read_csv(test_meta_path)

        test_corpus = load_corpus(self.config, df_test_meta, mode="test")

        return test_corpus
