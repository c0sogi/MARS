import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.vocabulary import CharVocab


class TextNormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    Handles loading, preprocessing, and caching of data.
    """

    def __init__(
        self,
        data_path,
        vocab,
        is_test=False,
        load_cached_data=True,
        max_len=Config.max_len,
    ):
        """
        Args:
            data_path (str): Path to the raw CSV file (metadata).
            vocab (CharVocab): Vocabulary instance for encoding text.
            is_test (bool): If True, does not look for 'after' column.
            load_cached_data (bool): Whether to use cached parquet files.
            max_len (int): Maximum sequence length for truncation (optional safety).
        """
        self.vocab = vocab
        self.is_test = is_test
        self.max_len = max_len

        # Determine cache path
        filename = os.path.basename(data_path)
        cache_name = os.path.splitext(filename)[0] + "_processed.parquet"
        self.cache_path = os.path.join(Config.WORKING_DIR, cache_name)

        # Load data
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached dataset from {self.cache_path}...")
            self.data = pd.read_parquet(self.cache_path)
        else:
            print(f"Processing dataset from {data_path}...")
            self.data = self._process_and_cache(data_path)

    def _process_and_cache(self, data_path):
        """
        Reads raw CSV, encodes text to indices, and saves to Parquet.
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Load raw data
        # We need 'id', 'before', and optionally 'after'
        cols = ["id", "before"]
        if not self.is_test:
            cols.append("after")

        try:
            df = pd.read_csv(data_path, usecols=cols, dtype=str)
        except ValueError:
            # Fallback if specific columns missing (e.g. if 'id' not in test meta)
            df = pd.read_csv(data_path, dtype=str)
            # Ensure ID exists
            if (
                "id" not in df.columns
                and "sentence_id" in df.columns
                and "token_id" in df.columns
            ):
                df["id"] = df["sentence_id"] + "_" + df["token_id"]

        # Fill NaNs
        df["before"] = df["before"].fillna("")
        if not self.is_test:
            df["after"] = df["after"].fillna("")

        # Helper for encoding
        def encode_text(text):
            # Encode with SOS and EOS
            indices = self.vocab.encode(text, add_special_tokens=True)
            # Truncate if necessary (though dynamic padding in collate is preferred)
            # We keep full length here, collate handles batch-level padding.
            # But we can enforce a hard limit to save memory if needed.
            if len(indices) > self.max_len:
                indices = indices[: self.max_len - 1] + [self.vocab.eos_idx]
            return indices

        # Apply encoding
        # Using apply is slow for huge datasets, but simple.
        # For 7M rows, this might take a minute or two.
        print("Encoding 'before' text...")
        df["src_indices"] = df["before"].apply(encode_text)

        if not self.is_test:
            print("Encoding 'after' text...")
            df["tgt_indices"] = df["after"].apply(encode_text)

        # Save to Parquet
        # Parquet supports columns of lists (arrays)
        print(f"Saving processed dataset to {self.cache_path}...")
        df.to_parquet(self.cache_path, index=False)

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'id': str,
                'src': torch.Tensor,
                'tgt': torch.Tensor (optional),
                'src_len': int
            }
        """
        row = self.data.iloc[idx]

        # Convert lists to tensors
        # Note: Parquet might load lists as numpy arrays or python lists
        src_indices = row["src_indices"]
        if isinstance(src_indices, np.ndarray):
            src_indices = src_indices.tolist()

        src_tensor = torch.tensor(src_indices, dtype=torch.long)

        item = {"id": row["id"], "src": src_tensor, "src_len": len(src_indices)}

        if not self.is_test:
            tgt_indices = row["tgt_indices"]
            if isinstance(tgt_indices, np.ndarray):
                tgt_indices = tgt_indices.tolist()
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)
            item["tgt"] = tgt_tensor

        return item


def collate_fn(batch):
    """
    Custom collate function to pad sequences.

    Args:
        batch (list): List of items from __getitem__.

    Returns:
        dict: Batch dictionary with padded tensors.
    """
    # Filter out None items if any
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    ids = [item["id"] for item in batch]
    src_lens = torch.tensor([item["src_len"] for item in batch], dtype=torch.long)

    # Pad source sequences
    src_tensors = [item["src"] for item in batch]
    # Assuming pad_idx is 0 (Config doesn't explicitly state it, but Vocabulary does)
    # We need to access vocab pad_idx. Since collate is a standalone function,
    # we assume standard 0 or pass it.
    # Vocabulary.py says self.pad_idx = 0.
    pad_idx = 0

    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx)

    batch_dict = {"id": ids, "src": src_padded, "src_len": src_lens}

    # Pad target sequences if they exist
    if "tgt" in batch[0]:
        tgt_tensors = [item["tgt"] for item in batch]
        tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_idx)
        batch_dict["tgt"] = tgt_padded

    return batch_dict


def get_dataloader(
    data_path,
    vocab,
    batch_size,
    is_test=False,
    shuffle=True,
    num_workers=4,
    load_cached_data=True,
):
    """
    Factory function to create a DataLoader.

    Args:
        data_path (str): Path to metadata CSV.
        vocab (CharVocab): Vocabulary object.
        batch_size (int): Batch size.
        is_test (bool): Whether this is a test set.
        shuffle (bool): Whether to shuffle data.
        num_workers (int): Number of subprocesses for loading.
        load_cached_data (bool): Whether to use caching.

    Returns:
        DataLoader
    """
    dataset = TextNormalizationDataset(
        data_path=data_path,
        vocab=vocab,
        is_test=is_test,
        load_cached_data=load_cached_data,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return loader
