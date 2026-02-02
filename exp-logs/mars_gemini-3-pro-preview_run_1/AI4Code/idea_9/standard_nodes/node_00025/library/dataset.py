import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class CachedNotebookDataset(Dataset):
    def __init__(self, split="train", debug=False):
        """
        Dataset class that loads precomputed embeddings from Parquet files.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            debug (bool): If True, limits the dataset size for debugging.
        """
        self.split = split
        self.debug = debug

        # Determine file path based on split
        if split == "train":
            self.file_path = Config.TRAIN_FEATURES_PATH
        elif split == "validation":
            self.file_path = Config.VAL_FEATURES_PATH
        elif split == "test":
            self.file_path = Config.TEST_FEATURES_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Features file not found at {self.file_path}. Please run preprocessing first."
            )

        print(f"Loading {split} features from {self.file_path}...")

        # Load Parquet file
        # The 'embedding' column contains arrays/lists.
        df = pd.read_parquet(self.file_path)

        if self.debug:
            print("Debug mode: limiting dataset to first 100 notebooks.")
            # Get first 100 unique IDs
            unique_ids = df["id"].unique()[:100]
            df = df[df["id"].isin(unique_ids)].copy()

        # Optimize memory layout for fast access
        # We group by ID to get indices for each notebook
        print("Indexing notebooks...")

        # Ensure data is sorted by ID and Rank (rank is -1 for test, but sorting by ID keeps cells together)
        # For Train/Val, sorting by rank ensures code cells are in correct order.
        if split != "test":
            df = df.sort_values(by=["id", "rank"])
        else:
            # For test, order is arbitrary but we group by ID
            df = df.sort_values(by=["id"])

        # Create numpy arrays for fast slicing
        # Stack embeddings into a single large float32 array
        self.embeddings = np.stack(df["embedding"].values).astype(np.float32)
        self.cell_types = df["cell_type"].values  # numpy array of strings
        self.ranks = df["rank"].values.astype(np.int32)

        # Create a lookup for notebook boundaries
        # groupby('id').indices returns a dict {id: array_of_indices}
        # Since we sorted by ID, the indices for a notebook are contiguous, but groupby indices might not be sorted if not careful.
        # However, df is sorted by ID.
        # To be safe and fast, we can use the fact that it's sorted.

        self.notebook_ids = df["id"].unique()

        # Map notebook ID to (start_index, count) or just the slice indices
        # Since we sorted, we can just store the indices directly from groupby
        self.notebook_indices = df.groupby("id", sort=False).indices

        print(
            f"Loaded {len(self.notebook_ids)} notebooks containing {len(self.embeddings)} cells."
        )

    def __len__(self):
        return len(self.notebook_ids)

    def __getitem__(self, idx):
        nb_id = self.notebook_ids[idx]
        indices = self.notebook_indices[nb_id]

        # Extract data for this notebook
        nb_embeddings = self.embeddings[indices]
        nb_types = self.cell_types[indices]
        nb_ranks = self.ranks[indices]

        # Separate Code and Markdown
        is_code = nb_types == "code"
        is_md = nb_types == "markdown"

        code_embeddings = nb_embeddings[is_code]
        md_embeddings = nb_embeddings[is_md]

        # For Train/Val, we need to generate labels
        # For Test, labels are dummy
        labels = []

        if self.split != "test":
            code_ranks = nb_ranks[is_code]
            md_ranks = nb_ranks[is_md]

            # Calculate Labels
            # Label = index of the code cell that immediately follows the markdown cell.
            # If MD is after all code cells, Label = len(code_cells) (which points to EOS token).

            # Since code_ranks are sorted (due to df sorting in __init__), we can use searchsorted
            if len(code_ranks) > 0:
                # np.searchsorted finds the first index where value > md_rank
                # side='right' with condition: we want the first code cell with rank > md_rank
                # code_ranks are ordered.
                # Example: Code Ranks [1, 3, 5]. MD Rank 2.
                # searchsorted([1, 3, 5], 2) -> index 1 (value 3). Correct.
                # MD Rank 6. searchsorted -> index 3 (len). Correct (EOS).
                # MD Rank 0. searchsorted -> index 0 (value 1). Correct.
                labels = np.searchsorted(code_ranks, md_ranks)
            else:
                # No code cells. All MD cells point to EOS (index 0)
                labels = np.zeros(len(md_ranks), dtype=np.int64)
        else:
            # Test set: no labels
            labels = np.full(len(md_embeddings), -100, dtype=np.int64)

        return {
            "id": nb_id,
            "code_embeddings": torch.tensor(code_embeddings, dtype=torch.float32),
            "markdown_embeddings": torch.tensor(md_embeddings, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def custom_collate_fn(batch):
    """
    Collate function to pad variable-length sequences of code and markdown cells.

    Args:
        batch (list): List of dicts from __getitem__.

    Returns:
        dict: Batch dictionary with padded tensors and masks.
    """
    ids = [item["id"] for item in batch]

    # Extract tensors
    code_seqs = [item["code_embeddings"] for item in batch]
    md_seqs = [item["markdown_embeddings"] for item in batch]
    labels_seqs = [item["labels"] for item in batch]

    # Get lengths
    code_lens = torch.tensor([len(seq) for seq in code_seqs], dtype=torch.long)
    md_lens = torch.tensor([len(seq) for seq in md_seqs], dtype=torch.long)

    # Pad sequences
    # batch_first=True -> (Batch, Seq, Dim)
    # Code padding: 0.0 (masked out anyway)
    padded_code = pad_sequence(code_seqs, batch_first=True, padding_value=0.0)

    # Markdown padding: 0.0
    padded_md = pad_sequence(md_seqs, batch_first=True, padding_value=0.0)

    # Label padding: -100 (standard ignore_index for CrossEntropyLoss)
    padded_labels = pad_sequence(labels_seqs, batch_first=True, padding_value=-100)

    # Create Attention Masks
    # 1 for valid token, 0 for padding (standard for some implementations)
    # OR True for padding (standard for PyTorch Transformer src_key_padding_mask)
    # We will provide boolean masks where True indicates PADDING (to be ignored).

    batch_size = len(batch)
    max_code_len = padded_code.size(1)
    max_md_len = padded_md.size(1)

    # Initialize with True (all padding)
    code_padding_mask = torch.ones((batch_size, max_code_len), dtype=torch.bool)
    md_padding_mask = torch.ones((batch_size, max_md_len), dtype=torch.bool)

    for i, length in enumerate(code_lens):
        code_padding_mask[i, :length] = False  # Valid positions are False

    for i, length in enumerate(md_lens):
        md_padding_mask[i, :length] = False

    return {
        "id": ids,
        "code_embeddings": padded_code,  # (B, Max_Code, 768)
        "code_lens": code_lens,  # (B,)
        "code_padding_mask": code_padding_mask,  # (B, Max_Code) - True where padding
        "markdown_embeddings": padded_md,  # (B, Max_MD, 768)
        "md_lens": md_lens,  # (B,)
        "md_padding_mask": md_padding_mask,  # (B, Max_MD)
        "labels": padded_labels,  # (B, Max_MD)
    }
