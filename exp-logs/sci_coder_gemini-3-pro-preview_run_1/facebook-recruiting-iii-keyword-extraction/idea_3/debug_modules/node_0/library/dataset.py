import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.vocabulary import Vocabulary


def collate_fn_offset(batch):
    """
    Custom collate function for processing variable length sequences with offsets.
    Suitable for EmbeddingBag and custom Attention Pooling layers.

    Args:
        batch: List of dictionaries/tuples returned by StackExchangeDataset.__getitem__

    Returns:
        dict: Batch dictionary containing flattened text tensors, offsets, and targets.
    """
    title_list = []
    body_list = []
    ids_list = []

    # Check if tags are present (Train/Val) or None (Test)
    has_tags = "tags" in batch[0] and batch[0]["tags"] is not None
    num_tags = batch[0]["num_classes"] if has_tags else 0

    # Initialize targets tensor if needed
    # We use a dense FloatTensor for BCEWithLogitsLoss
    if has_tags:
        targets = torch.zeros(len(batch), num_tags, dtype=torch.float32)
    else:
        targets = None

    for i, sample in enumerate(batch):
        title_list.append(sample["title"])
        body_list.append(sample["body"])
        ids_list.append(sample["id"])

        if has_tags:
            tag_indices = sample["tags"]
            if len(tag_indices) > 0:
                # Multi-hot encoding: set indices to 1.0
                targets[i, tag_indices] = 1.0

    # --- Process Title ---
    # Flatten list of lists into a single 1D list
    title_flat = [item for sublist in title_list for item in sublist]
    title_text = torch.tensor(title_flat, dtype=torch.long)

    # Calculate offsets: cumulative sum of lengths (starting with 0)
    # Example: lengths [2, 3] -> offsets [0, 2]
    title_lens = [len(x) for x in title_list]
    # We prepend 0, then take cumsum of lengths (excluding the last length for the offset list)
    title_offsets = torch.tensor([0] + title_lens[:-1], dtype=torch.long).cumsum(dim=0)

    # --- Process Body ---
    body_flat = [item for sublist in body_list for item in sublist]
    body_text = torch.tensor(body_flat, dtype=torch.long)

    body_lens = [len(x) for x in body_list]
    body_offsets = torch.tensor([0] + body_lens[:-1], dtype=torch.long).cumsum(dim=0)

    # --- Process IDs ---
    ids = torch.tensor(ids_list, dtype=torch.long)

    return {
        "title_text": title_text,
        "title_offsets": title_offsets,
        "body_text": body_text,
        "body_offsets": body_offsets,
        "targets": targets,
        "ids": ids,
    }


class StackExchangeDataset(Dataset):
    def __init__(self, split, vocab, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            vocab (Vocabulary): Initialized vocabulary object
            load_cached_data (bool): Whether to load from parquet cache
        """
        self.split = split
        self.vocab = vocab
        self.num_tags = vocab.get_num_tags()

        # Determine file paths based on split
        if split == "train":
            self.meta_path = Config.TRAIN_META
            self.cache_path = Config.CACHED_TRAIN
            self.data_csv = Config.TRAIN_CSV
        elif split == "val":
            self.meta_path = Config.VAL_META
            self.cache_path = Config.CACHED_VAL
            self.data_csv = Config.TRAIN_CSV  # Val is a subset of Train CSV
        elif split == "test":
            self.meta_path = Config.TEST_META
            self.cache_path = Config.CACHED_TEST
            self.data_csv = Config.TEST_CSV
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load Data (Cached or Built from scratch)
        self.data = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"[{self.split}] Loading cached data from {self.cache_path}...")
            try:
                df = pd.read_parquet(self.cache_path)
                return df
            except Exception as e:
                print(f"[{self.split}] Failed to load cache: {e}. Rebuilding...")

        # 2. Build from Scratch
        print(f"[{self.split}] Building dataset from scratch...")

        # Load Metadata to get IDs for this split
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata {self.meta_path} not found.")

        df_meta = pd.read_csv(self.meta_path)

        # Load Raw Text
        # We read specific columns to save memory.
        # Note: 'Tags' is in metadata for train/val, but we need Title/Body from source CSV.
        print(f"[{self.split}] Reading raw text from {self.data_csv}...")

        # Optimization: Read only necessary columns
        df_raw = pd.read_csv(
            self.data_csv,
            usecols=["Id", "Title", "Body"],
            dtype={"Id": "int64", "Title": "object", "Body": "object"},
        )

        # Merge Metadata with Raw Text on 'Id'
        # This filters df_raw to only include rows present in the current split
        print(f"[{self.split}] Merging metadata and raw text...")
        df_merged = df_meta.merge(df_raw, on="Id", how="inner")

        # Pre-process / Tokenize Text
        # We convert text to indices here to save CPU time during training loops
        print(f"[{self.split}] Tokenizing text (Title & Body)...")

        # Helper to tokenize a series
        def tokenize_series(series, max_len):
            # Use list comprehension for speed
            return [
                self.vocab.text_to_indices(text)[:max_len]
                for text in series.fillna("").astype(str)
            ]

        df_merged["title_idxs"] = tokenize_series(
            df_merged["Title"], Config.MAX_LEN_TITLE
        )
        df_merged["body_idxs"] = tokenize_series(df_merged["Body"], Config.MAX_LEN_BODY)

        # Encode Tags (if present)
        if "Tags" in df_merged.columns:
            print(f"[{self.split}] Encoding tags...")
            df_merged["tag_idxs"] = [
                self.vocab.tags_to_indices(tags)
                for tags in df_merged["Tags"].fillna("").astype(str)
            ]
        else:
            df_merged["tag_idxs"] = None

        # Select final columns for cache
        df_final = df_merged[["Id", "title_idxs", "body_idxs", "tag_idxs"]].copy()

        # Save to Cache
        print(f"[{self.split}] Saving cache to {self.cache_path}...")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df_final.to_parquet(self.cache_path, index=False)

        return df_final

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Retrieve indices. Parquet might return numpy arrays, convert to list for consistency
        title_idxs = row["title_idxs"]
        if isinstance(title_idxs, np.ndarray):
            title_idxs = title_idxs.tolist()

        body_idxs = row["body_idxs"]
        if isinstance(body_idxs, np.ndarray):
            body_idxs = body_idxs.tolist()

        item = {
            "id": row["Id"],
            "title": title_idxs,
            "body": body_idxs,
            "num_classes": self.num_tags,
        }

        # Handle Tags
        if row["tag_idxs"] is not None:
            tag_idxs = row["tag_idxs"]
            if isinstance(tag_idxs, np.ndarray):
                tag_idxs = tag_idxs.tolist()
            item["tags"] = tag_idxs
        else:
            item["tags"] = None

        return item


def get_loader(
    split,
    vocab,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create a DataLoader for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'
        vocab (Vocabulary): Vocabulary object
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes

    Returns:
        DataLoader: Configured PyTorch DataLoader
    """
    dataset = StackExchangeDataset(split, vocab)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_offset,
        pin_memory=Config.PIN_MEMORY,
    )

    return loader
