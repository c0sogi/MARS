import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class NotebookSequenceDataset(Dataset):
    """
    PyTorch Dataset for the Deep Sequence-Aware Position Regressor.

    Each sample corresponds to a single markdown cell (Query) and the sequence
    of code cells (Context) from the same notebook.

    Target: The relative rank of the markdown cell within the code sequence [0, 1].
    """

    def __init__(
        self, split: str, load_cached_data: bool = True, debug_limit: int = None
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed samples from disk cache.
            debug_limit (int): Limit number of notebooks for debugging.
        """
        self.split = split
        self.max_seq_len = Config.MAX_SEQ_LEN
        self.embed_dim = Config.EMBED_DIM

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Path for caching the processed list of samples
        # We cache the list of dictionaries to avoid re-grouping/processing pandas dataframes
        cache_filename = f"dataset_samples_{split}.pt"
        if debug_limit:
            cache_filename = f"dataset_samples_{split}_debug{debug_limit}.pt"
        self.cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading processed dataset from {self.cache_path}...")
            try:
                self.samples = torch.load(self.cache_path)
            except Exception as e:
                print(f"Failed to load dataset cache: {e}. Reprocessing...")
                self.samples = self._process_and_cache(debug_limit)
        else:
            self.samples = self._process_and_cache(debug_limit)

        print(f"Dataset ({split}) loaded with {len(self.samples)} samples.")

    def _process_and_cache(self, debug_limit):
        """
        Loads features from Parquet, processes them into samples, and saves to cache.
        """
        print(f"Processing dataset for split: {self.split}...")

        # Identify source file
        if self.split == "train":
            parquet_path = Config.TRAIN_CACHE_PATH
        elif self.split == "val":
            parquet_path = Config.VAL_CACHE_PATH
        elif self.split == "test":
            parquet_path = Config.TEST_CACHE_PATH
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Feature file not found: {parquet_path}. Run feature extraction first."
            )

        # Load Data
        df = pd.read_parquet(parquet_path)

        if debug_limit:
            # Filter by notebook_id to keep integrity of notebooks
            nb_ids = df["notebook_id"].unique()[:debug_limit]
            df = df[df["notebook_id"].isin(nb_ids)]

        samples = []

        # Helper to preserve order for test set
        # We assume the parquet file preserves the insertion order of rows
        df["orig_index"] = np.arange(len(df))

        # Group by notebook
        grouped = df.groupby("notebook_id")

        for nb_id, group in grouped:
            # Separate Code and Markdown
            code_df = group[group["cell_type"] == "code"]
            md_df = group[group["cell_type"] == "markdown"]

            # Determine Context Sequence (Code Cells)
            if self.split in ["train", "val"]:
                # For training/validation, we trust the 'rank' column derived from ground truth
                code_df = code_df.sort_values("rank")
            else:
                # For test, we trust the appearance order in the JSON (proxy: orig_index)
                code_df = code_df.sort_values("orig_index")

            # Extract embeddings
            if len(code_df) > 0:
                # Convert list of lists to numpy array
                # Note: df['embedding'] contains lists
                code_embeddings = np.stack(code_df["embedding"].values).astype(
                    np.float32
                )
                code_ranks = code_df["rank"].values
            else:
                code_embeddings = np.zeros((0, self.embed_dim), dtype=np.float32)
                code_ranks = np.array([])

            n_code = len(code_embeddings)

            # Create a sample for each markdown cell
            for _, row in md_df.iterrows():
                md_emb = np.array(row["embedding"], dtype=np.float32)
                cell_id = row["cell_id"]

                # Calculate Target
                target = 0.0
                if self.split in ["train", "val"]:
                    md_rank = row["rank"]
                    if n_code > 0:
                        # Target is the fraction of code cells that come BEFORE this markdown cell
                        # rank is global position.
                        # sum(code_rank < md_rank) counts code cells preceding this md
                        n_before = np.sum(code_ranks < md_rank)
                        target = float(n_before) / float(n_code)
                    else:
                        # Fallback for notebooks with no code cells
                        target = 0.5
                else:
                    # Dummy target for test
                    target = -1.0

                # Store sample
                # Note: We store a reference to the 'code_embeddings' array.
                # Since 'code_embeddings' is created once per notebook loop,
                # all markdown samples for this notebook share the same memory for context.
                samples.append(
                    {
                        "nb_id": nb_id,
                        "cell_id": cell_id,
                        "query": md_emb,
                        "context": code_embeddings,
                        "target": target,
                    }
                )

        # Save to cache
        print(f"Saving processed dataset to {self.cache_path}...")
        torch.save(samples, self.cache_path)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Query (Markdown)
        query = torch.tensor(sample["query"], dtype=torch.float32)

        # 2. Context (Code Sequence)
        context_raw = sample["context"]
        seq_len = len(context_raw)

        # Pad or Truncate
        if seq_len >= self.max_seq_len:
            # Truncate: take the first MAX_SEQ_LEN code cells
            # (Assumption: early code cells (imports, data load) are most informative)
            context_np = context_raw[: self.max_seq_len]
            mask_np = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            # Pad
            context_np = np.zeros((self.max_seq_len, self.embed_dim), dtype=np.float32)
            if seq_len > 0:
                context_np[:seq_len] = context_raw

            # Mask: 1 for Real, 0 for Pad
            mask_np = np.zeros(self.max_seq_len, dtype=np.float32)
            mask_np[:seq_len] = 1.0

        context = torch.tensor(context_np, dtype=torch.float32)
        mask = torch.tensor(mask_np, dtype=torch.float32)

        # 3. Label
        label = torch.tensor(sample["target"], dtype=torch.float32)

        # 4. IDs (for inference reconstruction)
        ids = (sample["nb_id"], sample["cell_id"])

        return {
            "query": query,
            "context": context,
            "mask": mask,
            "label": label,
            "ids": ids,
        }
