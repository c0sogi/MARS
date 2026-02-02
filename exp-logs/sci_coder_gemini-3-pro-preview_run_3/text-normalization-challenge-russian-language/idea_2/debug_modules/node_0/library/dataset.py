import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from library.config import Config
from library.utils import set_seed


class CharTokenizer:
    """
    Character-level tokenizer for Text Normalization.
    Handles vocabulary creation, saving/loading, and encoding/decoding.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

        # Initialize with special tokens from Config
        self.special_tokens = Config.SPECIAL_TOKENS
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token
        self.vocab_size = len(self.special_tokens)

    def fit(self, texts):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.char2idx[char] = idx
            self.idx2char[idx] = char

        self.vocab_size = len(self.char2idx)

    def encode(self, text):
        """
        Converts a string to a list of token indices.
        """
        text = str(text)
        return [self.char2idx.get(c, Config.UNK_IDX) for c in text]

    def decode(self, indices):
        """
        Converts a list of token indices back to a string.
        Ignores special tokens.
        """
        chars = []
        for idx in indices:
            if idx in [Config.PAD_IDX, Config.SOS_IDX, Config.EOS_IDX]:
                continue
            chars.append(self.idx2char.get(idx, Config.UNK_TOKEN))
        return "".join(chars)

    def save(self, path):
        """
        Saves the vocabulary to a .npy file.
        """
        np.save(path, self.char2idx)

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        self.char2idx = np.load(path, allow_pickle=True).item()
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)


def prepare_artifacts(load_cached_data=True):
    """
    Ensures that the vocabulary and class map exist.
    If not, generates them from the training data.

    Returns:
        tokenizer (CharTokenizer): Loaded or fitted tokenizer.
        class_map (dict): Loaded or created class mapping.
    """
    tokenizer = CharTokenizer()
    class_map = {}

    vocab_exists = os.path.exists(Config.VOCAB_FILE)
    class_map_exists = os.path.exists(Config.CLASS_MAP_FILE)

    if load_cached_data and vocab_exists and class_map_exists:
        tokenizer.load(Config.VOCAB_FILE)
        class_map = np.load(Config.CLASS_MAP_FILE, allow_pickle=True).item()
    else:
        # Load raw training data to build artifacts
        df_train = pd.read_csv(
            Config.TRAIN_CSV,
            dtype={
                Config.INPUT_COL: str,
                Config.TARGET_COL: str,
                Config.CLASS_COL: str,
            },
        )

        # Handle NaNs
        df_train[Config.INPUT_COL] = df_train[Config.INPUT_COL].fillna("")
        df_train[Config.TARGET_COL] = df_train[Config.TARGET_COL].fillna("")

        # Fit Tokenizer
        all_text = pd.concat([df_train[Config.INPUT_COL], df_train[Config.TARGET_COL]])
        tokenizer.fit(all_text)
        tokenizer.save(Config.VOCAB_FILE)

        # Fit Class Map
        unique_classes = sorted(df_train[Config.CLASS_COL].dropna().unique())
        class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
        np.save(Config.CLASS_MAP_FILE, class_map)

    return tokenizer, class_map


def process_data(split, tokenizer, class_map, load_cached_data=True):
    """
    Loads and processes data for a specific split.
    Uses caching (Parquet) to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        tokenizer (CharTokenizer): Fitted tokenizer.
        class_map (dict): Class to index mapping.
        load_cached_data (bool): Whether to use cached files.

    Returns:
        pd.DataFrame: Processed dataframe with tokenized columns.
    """
    # Determine file paths
    if split == "train":
        csv_path = Config.TRAIN_CSV
        parquet_path = Config.TRAIN_PROCESSED
    elif split == "val":
        csv_path = Config.VAL_CSV
        parquet_path = Config.VAL_PROCESSED
    elif split == "test":
        csv_path = Config.TEST_CSV
        parquet_path = Config.TEST_PROCESSED
    else:
        raise ValueError(f"Unknown split: {split}")

    # Check cache
    if load_cached_data and os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            return df
        except Exception:
            pass  # Fallback to processing

    # Load Raw Data
    # Specify dtypes to avoid mixed type warnings
    df = pd.read_csv(csv_path, dtype={"sentence_id": str, "token_id": str})

    # Debugging: Sample subset if configured
    if Config.DEBUG_SAMPLE_SIZE is not None and len(df) > Config.DEBUG_SAMPLE_SIZE:
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    # Handle NaNs in text
    if Config.INPUT_COL in df.columns:
        df[Config.INPUT_COL] = df[Config.INPUT_COL].fillna("")
    if Config.TARGET_COL in df.columns:
        df[Config.TARGET_COL] = df[Config.TARGET_COL].fillna("")
    if Config.CLASS_COL in df.columns:
        df[Config.CLASS_COL] = df[Config.CLASS_COL].fillna("PLAIN")  # Default fallback

    # Tokenize
    # We use apply to create lists of integers
    df["src_ids"] = df[Config.INPUT_COL].astype(str).apply(tokenizer.encode)

    if split != "test":
        df["tgt_ids"] = df[Config.TARGET_COL].astype(str).apply(tokenizer.encode)
        df["class_id"] = (
            df[Config.CLASS_COL].map(lambda x: class_map.get(x, 0)).astype(int)
        )
    else:
        # For test, we don't have targets or classes
        # Create dummy columns for consistency in Dataset class
        df["tgt_ids"] = None
        df["class_id"] = -1

    # Save to Parquet
    # Ensure working directory exists
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df.to_parquet(parquet_path)

    return df


class TextNormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    Returns:
        src: Encoder input (padded)
        tgt_in: Decoder input (SOS + sequence + padded)
        tgt_out: Decoder target (sequence + EOS + padded)
        class_id: Auxiliary class label
        src_len: Length of source sequence (for masking)
        tgt_len: Length of target sequence
    """

    def __init__(self, df, max_seq_len=Config.MAX_SEQ_LEN):
        self.df = df
        self.max_seq_len = max_seq_len
        self.pad_idx = Config.PAD_IDX
        self.sos_idx = Config.SOS_IDX
        self.eos_idx = Config.EOS_IDX

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Source Processing ---
        src_ids = row["src_ids"]
        src_len = len(src_ids)

        # Truncate if necessary
        if src_len > self.max_seq_len:
            src_ids = src_ids[: self.max_seq_len]
            src_len = self.max_seq_len

        # Pad Source
        src_tensor = torch.full((self.max_seq_len,), self.pad_idx, dtype=torch.long)
        src_tensor[:src_len] = torch.tensor(src_ids, dtype=torch.long)

        # --- Target Processing ---
        if row["tgt_ids"] is not None:
            tgt_ids = row["tgt_ids"]
            # Truncate (leave room for SOS/EOS)
            if len(tgt_ids) > self.max_seq_len - 1:
                tgt_ids = tgt_ids[: self.max_seq_len - 1]

            # Decoder Input: <SOS> + tokens
            tgt_in = [self.sos_idx] + tgt_ids
            tgt_in_len = len(tgt_in)
            tgt_in_tensor = torch.full(
                (self.max_seq_len,), self.pad_idx, dtype=torch.long
            )
            tgt_in_tensor[:tgt_in_len] = torch.tensor(tgt_in, dtype=torch.long)

            # Decoder Target: tokens + <EOS>
            tgt_out = tgt_ids + [self.eos_idx]
            tgt_out_len = len(tgt_out)
            tgt_out_tensor = torch.full(
                (self.max_seq_len,), self.pad_idx, dtype=torch.long
            )
            tgt_out_tensor[:tgt_out_len] = torch.tensor(tgt_out, dtype=torch.long)

            class_id = torch.tensor(row["class_id"], dtype=torch.long)
        else:
            # Inference mode (Test set)
            tgt_in_tensor = torch.full(
                (self.max_seq_len,), self.pad_idx, dtype=torch.long
            )
            tgt_in_tensor[0] = self.sos_idx  # Start with SOS
            tgt_out_tensor = torch.full(
                (self.max_seq_len,), self.pad_idx, dtype=torch.long
            )
            class_id = torch.tensor(-1, dtype=torch.long)

        return {
            "src": src_tensor,
            "tgt_in": tgt_in_tensor,
            "tgt_out": tgt_out_tensor,
            "class_id": class_id,
            "src_len": torch.tensor(src_len, dtype=torch.long),
            # Pass original IDs for submission file generation
            "id": row[Config.ID_COL] if Config.ID_COL in row else "",
        }


def get_weighted_dataloader(
    dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates a DataLoader with WeightedRandomSampler to handle class imbalance.
    """
    # Extract class labels from the dataframe
    # We assume dataset.df exists and has 'class_id'
    targets = dataset.df["class_id"].values

    # Calculate class counts
    class_counts = Counter(targets)

    # Calculate weight per class (inverse frequency)
    # Add small epsilon to avoid division by zero if a class is missing in the split (unlikely)
    class_weights = {cls: 1.0 / (count + 1e-6) for cls, count in class_counts.items()}

    # Assign weight to each sample
    sample_weights = [class_weights[t] for t in targets]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader


def get_dataloader(
    dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, shuffle=False
):
    """
    Standard DataLoader for validation and testing.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
