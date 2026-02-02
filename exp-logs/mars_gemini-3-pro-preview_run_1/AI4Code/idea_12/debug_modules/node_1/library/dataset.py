import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


class CachedNotebookDataset(Dataset):
    """
    Dataset class for loading pre-computed notebook embeddings from Parquet files.
    """

    def __init__(self, file_path, config=None):
        """
        Args:
            file_path (str): Path to the parquet file containing embeddings.
            config (Config): Configuration object.
        """
        self.config = config if config else Config
        self.file_path = file_path

        # Load data
        # We expect the parquet file to have columns:
        # id, code_embeddings, markdown_embeddings, code_ids, markdown_ids, markdown_labels
        try:
            self.df = pd.read_parquet(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find dataset file at {file_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Process Code Embeddings
        # Stored as list of lists in parquet
        code_emb_list = row["code_embeddings"]
        if len(code_emb_list) > 0:
            code_emb = torch.tensor(code_emb_list, dtype=torch.float)
        else:
            # Handle empty code cells: shape [0, input_dim]
            code_emb = torch.empty((0, self.config.INPUT_DIM), dtype=torch.float)

        # 2. Process Markdown Embeddings
        md_emb_list = row["markdown_embeddings"]
        if len(md_emb_list) > 0:
            md_emb = torch.tensor(md_emb_list, dtype=torch.float)
        else:
            md_emb = torch.empty((0, self.config.INPUT_DIM), dtype=torch.float)

        # 3. Process Labels
        # Labels represent the index of the code cell immediately following the markdown cell.
        # Range: 0 to N_code. (N_code implies it's after the last code cell).
        labels_list = row["markdown_labels"]
        if len(labels_list) > 0:
            labels = torch.tensor(labels_list, dtype=torch.long)
        else:
            labels = torch.empty((0,), dtype=torch.long)

        return {
            "id": row["id"],
            "code_emb": code_emb,
            "md_emb": md_emb,
            "labels": labels,
            "code_ids": row["code_ids"],
            "md_ids": row["markdown_ids"],
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.

    Returns:
        dict: containing:
            - code_emb: [B, Max_Code_Len, Dim]
            - code_lens: [B] (Original lengths, used for EOS insertion)
            - md_emb: [B, Max_MD_Len, Dim]
            - md_mask: [B, Max_MD_Len] (True for valid tokens, False for padding)
            - labels: [B, Max_MD_Len] (Padded with -100)
            - ids: List of notebook IDs
            - code_ids: List of lists of code cell IDs
            - md_ids: List of lists of markdown cell IDs
    """
    ids = [item["id"] for item in batch]
    code_ids = [item["code_ids"] for item in batch]
    md_ids = [item["md_ids"] for item in batch]

    # Extract tensors
    code_seqs = [item["code_emb"] for item in batch]
    md_seqs = [item["md_emb"] for item in batch]
    label_seqs = [item["labels"] for item in batch]

    # 1. Pad Code Sequences
    # We need the lengths to insert the EOS token dynamically in the model
    code_lens = torch.tensor([len(s) for s in code_seqs], dtype=torch.long)

    # Pad code embeddings.
    # Note: We don't strictly need a mask for the Transformer Encoder if we use src_key_padding_mask,
    # but the model logic described relies on indices (code_lens) to place EOS.
    # Padding value 0.0 is standard.
    if len(code_seqs) > 0:
        padded_code_emb = pad_sequence(code_seqs, batch_first=True, padding_value=0.0)
    else:
        padded_code_emb = torch.empty(
            0, 0, 768
        )  # Should not happen with valid batch size

    # 2. Pad Markdown Sequences (Set)
    md_lens = torch.tensor([len(s) for s in md_seqs], dtype=torch.long)

    if len(md_seqs) > 0:
        padded_md_emb = pad_sequence(md_seqs, batch_first=True, padding_value=0.0)
    else:
        padded_md_emb = torch.empty(0, 0, 768)

    # Create Mask for Markdown (True for valid, False for padding)
    # Shape: [B, Max_MD_Len]
    batch_size = len(batch)
    max_md_len = padded_md_emb.size(1) if padded_md_emb.dim() > 1 else 0

    md_mask = torch.zeros((batch_size, max_md_len), dtype=torch.bool)
    for i, length in enumerate(md_lens):
        if length > 0:
            md_mask[i, :length] = True

    # 3. Pad Labels
    # Padding value -100 is ignored by CrossEntropyLoss
    if len(label_seqs) > 0:
        padded_labels = pad_sequence(label_seqs, batch_first=True, padding_value=-100)
    else:
        padded_labels = torch.empty(0, 0, dtype=torch.long)

    return {
        "code_emb": padded_code_emb,
        "code_lens": code_lens,
        "md_emb": padded_md_emb,
        "md_mask": md_mask,
        "labels": padded_labels,
        "ids": ids,
        "code_ids": code_ids,
        "md_ids": md_ids,
    }
