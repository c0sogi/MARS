import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import CharTokenizer


def load_metadata(split="train"):
    """
    Loads the raw metadata parquet files.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if split == "train":
        path = Config.TRAIN_DATA
    elif split == "val":
        path = Config.VAL_DATA
    elif split == "test":
        path = Config.TEST_DATA
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_parquet(path)


def process_data(split="train", load_cached_data=True):
    """
    Loads metadata, adds context (prev/next tokens), filters for neural training
    (if train/val), and caches the result.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'prev', 'next' columns.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_processed.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading processed {split} data from {cache_path}...")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data...")
    df = load_metadata(split)

    # Ensure sorting for correct context extraction
    # We assume 'sentence_id' and 'token_id' exist.
    if "sentence_id" in df.columns and "token_id" in df.columns:
        df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

    # Fill NA in text columns to avoid string errors
    df["before"] = df["before"].fillna("")
    if "after" in df.columns:
        df["after"] = df["after"].fillna("")

    # Vectorized Context Extraction
    # Shift 'before' column to get prev and next
    # We must respect sentence boundaries.
    print("Generating context columns...")

    # Create shifted series
    prev_series = df["before"].shift(1).fillna("")
    next_series = df["before"].shift(-1).fillna("")

    # Check sentence boundaries
    # prev is valid if current sentence_id == prev sentence_id
    # next is valid if current sentence_id == next sentence_id
    sent_ids = df["sentence_id"]
    prev_sent_ids = sent_ids.shift(1)
    next_sent_ids = sent_ids.shift(-1)

    # Apply masks
    # If boundary change, context is empty string
    is_same_prev = sent_ids == prev_sent_ids
    is_same_next = sent_ids == next_sent_ids

    df["prev"] = np.where(is_same_prev, prev_series, "")
    df["next"] = np.where(is_same_next, next_series, "")

    # Filter for Neural Training
    # We only filter train and val sets. Test set must keep all rows (or be filtered by inference logic later).
    # The requirement says: "Create a specialized training set... by excluding PLAIN or PUNCT"
    if split in ["train", "val"]:
        print(f"Filtering {split} data (removing PLAIN and PUNCT)...")
        initial_len = len(df)
        df = df[~df["class"].isin(["PLAIN", "PUNCT"])].copy()
        print(f"Filtered {split}: {initial_len} -> {len(df)} rows")

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Saving processed {split} data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def build_tokenizer(load_cached_data=True):
    """
    Initializes and builds the vocabulary for the CharTokenizer.
    Uses the full training set (before and after text) to ensure coverage.
    """
    tokenizer = CharTokenizer()

    # If cache exists for vocab (handled inside tokenizer), we can skip data loading if we trust it.
    # However, tokenizer.build_vocab checks its own cache.
    # We need to provide texts if cache doesn't exist.

    vocab_cache = os.path.join(Config.WORKING_DIR, "vocab.json")
    if load_cached_data and os.path.exists(vocab_cache):
        tokenizer.build_vocab([], load_cached_data=True)
        return tokenizer

    # Load raw train data to build full vocab
    print("Loading training data for vocabulary building...")
    df_train = load_metadata("train")

    # Collect all text
    texts = (
        df_train["before"].astype(str).tolist() + df_train["after"].astype(str).tolist()
    )

    tokenizer.build_vocab(texts, load_cached_data=load_cached_data)
    return tokenizer


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for Seq2Seq Text Normalization.
    Constructs input: [prev] <SEP> [curr] <SEP> [next]
    """

    def __init__(self, df, tokenizer, max_len=Config.MAX_SEQ_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sep_token = Config.SEP_TOKEN

        # Pre-convert columns to list for faster access in __getitem__
        self.before = self.df["before"].astype(str).tolist()
        self.prev = self.df["prev"].astype(str).tolist()
        self.next = self.df["next"].astype(str).tolist()

        # Check if targets exist
        self.has_target = "after" in self.df.columns
        if self.has_target:
            self.after = self.df["after"].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct Contextual Input
        # Format: prev <SEP> curr <SEP> next
        curr_text = self.before[idx]
        prev_text = self.prev[idx]
        next_text = self.next[idx]

        # We can limit context length if needed, but char level usually fits.
        # Simple concatenation
        input_str = f"{prev_text}{self.sep_token}{curr_text}{self.sep_token}{next_text}"

        # Encode Input
        # We do NOT add special tokens (SOS/EOS) to encoder input usually,
        # but for some Seq2Seq it helps. Config doesn't specify strictly,
        # but usually Encoder input: raw indices, Decoder input: SOS...EOS.
        # Let's stick to standard: Encoder gets indices.
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)

        # Truncate if necessary (rare for char level single tokens, but context adds up)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "raw_text": curr_text,
        }

        # Encode Target if available
        if self.has_target:
            target_text = self.after[idx]
            # Decoder targets need SOS and EOS
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=True)

            if len(target_ids) > self.max_len:
                target_ids = target_ids[: self.max_len]

            result["target_ids"] = torch.tensor(target_ids, dtype=torch.long)
            result["target_text"] = target_text

        return result


def collate_fn(batch):
    """
    Custom collate function to pad sequences.
    """
    input_ids = [item["input_ids"] for item in batch]

    # Pad inputs
    # Use PAD_IDX from Config
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=Config.PAD_IDX
    )

    batch_out = {
        "input_ids": input_ids_padded,
        "raw_text": [item["raw_text"] for item in batch],
    }

    # Create attention mask (1 for real token, 0 for pad)
    attention_mask = (input_ids_padded != Config.PAD_IDX).long()
    batch_out["attention_mask"] = attention_mask

    if "target_ids" in batch[0]:
        target_ids = [item["target_ids"] for item in batch]
        target_ids_padded = pad_sequence(
            target_ids, batch_first=True, padding_value=Config.PAD_IDX
        )
        batch_out["target_ids"] = target_ids_padded
        batch_out["target_text"] = [item["target_text"] for item in batch]

    return batch_out


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_sample_size=None
):
    """
    High-level function to get train and validation dataloaders.

    Args:
        batch_size (int): Batch size.
        load_cached_data (bool): Use caching.
        debug_sample_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    # 1. Build/Load Tokenizer
    tokenizer = build_tokenizer(load_cached_data=load_cached_data)

    # 2. Process Data
    df_train = process_data("train", load_cached_data=load_cached_data)
    df_val = process_data("val", load_cached_data=load_cached_data)

    # Debugging: Subsample
    if debug_sample_size:
        print(f"DEBUG: Subsampling datasets to {debug_sample_size} samples.")
        if len(df_train) > debug_sample_size:
            df_train = df_train.iloc[:debug_sample_size]
        if len(df_val) > debug_sample_size:
            df_val = df_val.iloc[:debug_sample_size]

    # 3. Create Datasets
    train_dataset = NormalizationDataset(df_train, tokenizer)
    val_dataset = NormalizationDataset(df_val, tokenizer)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() or 4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() or 4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, tokenizer
