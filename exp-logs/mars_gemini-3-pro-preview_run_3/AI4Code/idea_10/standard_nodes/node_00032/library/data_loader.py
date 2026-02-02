import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from library.config import Config
from library.utils import read_notebook, set_seed


class RelaxedProximityDataset(Dataset):
    """
    Dataset for contrastive fine-tuning of the semantic backbone.
    Generates positive pairs (Markdown, Nearest Subsequent Code) to learn
    structural proximity.
    """

    def __init__(self, metadata_path, load_cached_data=True, subset_size=None):
        """
        Args:
            metadata_path (str): Path to the train metadata CSV.
            load_cached_data (bool): If True, attempts to load pre-computed pairs from disk.
            subset_size (int, optional): Number of notebooks to sample for fine-tuning.
                                         Defaults to Config.FINE_TUNE_SUBSET_SIZE.
        """
        self.metadata_path = metadata_path
        self.subset_size = (
            subset_size if subset_size is not None else Config.FINE_TUNE_SUBSET_SIZE
        )
        self.cache_path = Config.TRAIN_PAIRS_PATH

        # Load data (either from cache or by generating it)
        self.pairs = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached pairs from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                return df.to_dict("records")
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")

        # 2. Generate from scratch if cache missing or load failed
        return self._generate_pairs()

    def _generate_pairs(self):
        print(f"Generating training pairs from {self.subset_size} notebooks...")
        set_seed(Config.SEED)

        # Load metadata
        df_meta = pd.read_csv(self.metadata_path)

        # Sample notebooks to fit within runtime constraints
        if len(df_meta) > self.subset_size:
            df_meta = df_meta.sample(
                n=self.subset_size, random_state=Config.SEED
            ).reset_index(drop=True)

        all_pairs = []

        # Iterate through sampled notebooks
        for _, row in df_meta.iterrows():
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            # cell_order is a space-delimited string of cell IDs
            cell_order = str(row["cell_order"]).split()

            try:
                code_cells, md_cells = read_notebook(file_path)
            except Exception:
                continue

            if not code_cells or not md_cells:
                continue

            # Create lookup dictionaries for fast access
            code_dict = {c["id"]: c["source"] for c in code_cells}
            md_dict = {c["id"]: c["source"] for c in md_cells}

            # Reconstruct the ordered sequence of (type, source)
            ordered_cells = []
            for cid in cell_order:
                if cid in code_dict:
                    ordered_cells.append(("code", code_dict[cid]))
                elif cid in md_dict:
                    ordered_cells.append(("markdown", md_dict[cid]))

            # Pair Generation: Markdown -> Nearest Subsequent Code
            # We iterate forward. Markdown cells are buffered until a Code cell is found.
            md_buffer = []
            for c_type, source in ordered_cells:
                if c_type == "markdown":
                    md_buffer.append(source)
                elif c_type == "code":
                    # Pair all buffered markdowns with this code cell
                    for md_text in md_buffer:
                        # Basic filtering: skip empty cells
                        if md_text.strip() and source.strip():
                            all_pairs.append({"markdown": md_text, "code": source})
                    # Clear buffer after pairing
                    md_buffer = []

            # Note: Markdown cells appearing after the last code cell are dropped
            # as they have no "subsequent" code cell to pair with.

        # Save generated pairs to cache
        print(f"Generated {len(all_pairs)} pairs. Saving to {self.cache_path}...")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df_pairs = pd.DataFrame(all_pairs)
        df_pairs.to_parquet(self.cache_path, index=False)

        return all_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns an InputExample compatible with sentence-transformers.
        """
        item = self.pairs[idx]
        return InputExample(texts=[item["markdown"], item["code"]])


def get_notebook_iterator(metadata_path):
    """
    Generator that yields notebook data for feature extraction.

    Args:
        metadata_path (str): Path to the metadata CSV.

    Yields:
        tuple: (notebook_id, code_cells, markdown_cells)
    """
    df = pd.read_csv(metadata_path)

    for _, row in df.iterrows():
        nb_id = row["id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            code_cells, md_cells = read_notebook(file_path)
            yield nb_id, code_cells, md_cells
        except Exception as e:
            # Skip corrupted notebooks silently or with minimal log
            continue
