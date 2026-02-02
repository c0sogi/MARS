import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.preprocess import FeatureExtractor


class CachedDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        Dataset class that loads pre-computed embeddings from Parquet files.
        Handles automatic generation of features if cache is missing.
        """
        self.config = Config()
        self.mode = mode

        # Resolve paths based on mode
        if mode == "train":
            self.parquet_path = self.config.TRAIN_FEATS_PATH
            self.metadata_path = self.config.TRAIN_METADATA_PATH
        elif mode == "val":
            self.parquet_path = self.config.VAL_FEATS_PATH
            self.metadata_path = self.config.VAL_METADATA_PATH
        elif mode == "test":
            self.parquet_path = self.config.TEST_FEATS_PATH
            self.metadata_path = self.config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # Ensure working directory exists
        os.makedirs(os.path.dirname(self.parquet_path), exist_ok=True)

        # Logic: Load cache if requested and exists, else process from scratch
        if load_cached_data and os.path.exists(self.parquet_path):
            self.df = pd.read_parquet(self.parquet_path)
        else:
            extractor = FeatureExtractor()
            # process_dataset saves to disk and returns the dataframe
            self.df = extractor.process_dataset(
                self.metadata_path, self.parquet_path, load_cached_data=False
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract embeddings
        # Parquet stores lists; convert to float32 numpy arrays first for efficiency
        code_embs = np.array(row["code_embeddings"], dtype=np.float32)
        md_embs = np.array(row["markdown_embeddings"], dtype=np.float32)

        # Extract labels
        # Handle cases where labels might be None (test set) or empty
        labels = row.get("markdown_labels", [])
        if labels is None:
            labels = []
        labels = np.array(labels, dtype=np.int64)

        # If labels are empty but we have markdown cells (e.g. test set default), fill with -1
        if len(labels) == 0 and len(md_embs) > 0:
            labels = np.full(len(md_embs), -1, dtype=np.int64)

        # --- Truncation Logic ---
        # Enforce MAX_SEQ_LEN to prevent OOM on extremely large notebooks.
        # We apply this independently to code and markdown streams.
        max_seq_len = self.config.MAX_SEQ_LEN

        # Truncate Code Sequence
        if len(code_embs) > max_seq_len:
            code_embs = code_embs[:max_seq_len]
            # If code is truncated, labels pointing to indices > new_len are invalid.
            # We clip them to len(code_embs), which represents the new EOS position.
            labels = np.clip(labels, 0, len(code_embs))

        # Truncate Markdown Sequence
        if len(md_embs) > max_seq_len:
            md_embs = md_embs[:max_seq_len]
            labels = labels[:max_seq_len]

        return {
            "id": row["id"],
            "code_embeddings": torch.tensor(code_embs),
            "markdown_embeddings": torch.tensor(md_embs),
            "labels": torch.tensor(labels),
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads code and markdown embeddings to the maximum length in the batch.
    Returns valid lengths for dynamic EOS insertion and masking.
    """
    # 1. Determine max lengths in this batch
    max_code_len = 0
    max_md_len = 0

    for item in batch:
        max_code_len = max(max_code_len, item["code_embeddings"].size(0))
        max_md_len = max(max_md_len, item["markdown_embeddings"].size(0))

    # Handle rare edge case of empty batch/sequences
    if max_code_len == 0:
        max_code_len = 1
    if max_md_len == 0:
        max_md_len = 1

    batch_size = len(batch)
    # Infer embedding dimension from the first non-empty item, default to 768
    emb_dim = 768
    for item in batch:
        if item["code_embeddings"].numel() > 0:
            emb_dim = item["code_embeddings"].size(1)
            break

    # 2. Allocate padded tensors
    # Code: (B, Max_Code, Dim)
    code_padded = torch.zeros(batch_size, max_code_len, emb_dim)

    # Markdown: (B, Max_MD, Dim)
    md_padded = torch.zeros(batch_size, max_md_len, emb_dim)

    # Labels: (B, Max_MD) - Fill with -100 (CrossEntropyLoss ignore_index)
    labels_padded = torch.full((batch_size, max_md_len), -100, dtype=torch.long)

    # Metadata containers
    code_lens = []
    md_lens = []
    ids = []

    # 3. Fill tensors
    for i, item in enumerate(batch):
        c_emb = item["code_embeddings"]
        m_emb = item["markdown_embeddings"]
        lbl = item["labels"]

        c_len = c_emb.size(0)
        m_len = m_emb.size(0)

        if c_len > 0:
            code_padded[i, :c_len, :] = c_emb
        if m_len > 0:
            md_padded[i, :m_len, :] = m_emb
            labels_padded[i, :m_len] = lbl

        code_lens.append(c_len)
        md_lens.append(m_len)
        ids.append(item["id"])

    return {
        "code_embeddings": code_padded,
        "markdown_embeddings": md_padded,
        "labels": labels_padded,
        "code_lens": torch.tensor(code_lens, dtype=torch.long),
        "md_lens": torch.tensor(md_lens, dtype=torch.long),
        "ids": ids,
    }
