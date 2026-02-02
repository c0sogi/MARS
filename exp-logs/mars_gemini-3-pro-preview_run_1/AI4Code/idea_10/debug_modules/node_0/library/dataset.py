import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.preprocess import FeatureExtractor


class CachedEmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed embeddings (Code & Markdown) from Parquet files.
    Handles dynamic generation if files are missing.
    """

    def __init__(self, features_path, split_name="train"):
        """
        Args:
            features_path (str): Path to the parquet file.
            split_name (str): One of 'train', 'val', 'test'.
        """
        self.split_name = split_name
        self.features_path = features_path

        # Check if features exist; if not, generate them
        if not os.path.exists(features_path):
            print(
                f"[{split_name}] Features not found at {features_path}. Generating..."
            )
            extractor = FeatureExtractor()

            if split_name == "train":
                meta_path = Config.TRAIN_METADATA_PATH
                is_test = False
            elif split_name == "val":
                meta_path = Config.VAL_METADATA_PATH
                is_test = False
            elif split_name == "test":
                meta_path = Config.TEST_METADATA_PATH
                is_test = True
            else:
                raise ValueError(f"Unknown split name: {split_name}")

            self.df = extractor.process_dataset(
                meta_path, features_path, load_cached_data=True, is_test=is_test
            )
        else:
            self.df = pd.read_parquet(features_path, engine="pyarrow")

        # Debugging: Subsample if configured
        if Config.DEBUG_SAMPLE_SIZE is not None:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)
            print(f"[{split_name}] Debug mode: Loaded {len(self.df)} samples.")
        else:
            print(f"[{split_name}] Loaded {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Process Code Embeddings ---
        code_arr = row["code_embeddings"]
        # Handle empty or malformed arrays
        if code_arr is None or len(code_arr) == 0:
            code_emb = torch.empty((0, 768), dtype=torch.float)
        else:
            code_emb = torch.from_numpy(code_arr).float()
            # Ensure 2D shape (Seq, Dim)
            if code_emb.dim() == 1:
                if code_emb.size(0) == 0:
                    code_emb = code_emb.view(0, 768)
                else:
                    # If somehow we got a single vector without batch dim
                    code_emb = code_emb.unsqueeze(0)

        # --- Process Markdown Embeddings & Labels ---
        md_arr = row["markdown_embeddings"]
        if md_arr is None or len(md_arr) == 0:
            md_emb = torch.empty((0, 768), dtype=torch.float)
            labels = torch.empty((0,), dtype=torch.long)
        else:
            md_emb = torch.from_numpy(md_arr).float()
            if md_emb.dim() == 1:
                if md_emb.size(0) == 0:
                    md_emb = md_emb.view(0, 768)
                else:
                    md_emb = md_emb.unsqueeze(0)

            labels = torch.from_numpy(row["markdown_labels"]).long()

        return {
            "id": row["id"],
            "code_embeddings": code_emb,
            "markdown_embeddings": md_emb,
            "labels": labels,
            "num_code": code_emb.size(0),
            "num_md": md_emb.size(0),
        }


def collate_fn(batch):
    """
    Collates a batch of notebooks with variable numbers of cells.
    Pads sequences and generates attention masks.
    """
    ids = [item["id"] for item in batch]

    code_embs = [item["code_embeddings"] for item in batch]
    md_embs = [item["markdown_embeddings"] for item in batch]
    labels = [item["labels"] for item in batch]

    code_lens = [item["num_code"] for item in batch]
    md_lens = [item["num_md"] for item in batch]

    # Pad sequences (Batch, MaxSeq, Dim)
    # padding_value=0.0 is standard for embeddings
    padded_code_emb = pad_sequence(code_embs, batch_first=True, padding_value=0.0)
    padded_md_emb = pad_sequence(md_embs, batch_first=True, padding_value=0.0)

    # Pad labels with -100 (ignore index)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Generate Masks
    # True indicates value should be IGNORED (padding)
    batch_size = len(batch)
    max_code_len = padded_code_emb.size(1)
    max_md_len = padded_md_emb.size(1)

    code_mask = torch.ones((batch_size, max_code_len), dtype=torch.bool)
    md_mask = torch.ones((batch_size, max_md_len), dtype=torch.bool)

    for i, (c_len, m_len) in enumerate(zip(code_lens, md_lens)):
        if c_len > 0:
            code_mask[i, :c_len] = False
        if m_len > 0:
            md_mask[i, :m_len] = False

    return {
        "id": ids,
        "code_embeddings": padded_code_emb,
        "markdown_embeddings": padded_md_emb,
        "labels": padded_labels,
        "code_mask": code_mask,
        "markdown_mask": md_mask,
        "code_lens": torch.tensor(code_lens, dtype=torch.long),
        "markdown_lens": torch.tensor(md_lens, dtype=torch.long),
    }


def get_dataloaders(batch_size=None, num_workers=None):
    """
    Factory function to create Train, Val, and Test dataloaders.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Instantiate Datasets
    train_ds = CachedEmbeddingDataset(Config.TRAIN_FEATURES_PATH, split_name="train")
    val_ds = CachedEmbeddingDataset(Config.VAL_FEATURES_PATH, split_name="val")
    test_ds = CachedEmbeddingDataset(Config.TEST_FEATURES_PATH, split_name="test")

    # Create Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
