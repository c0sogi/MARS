import os
import pandas as pd
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import is_semiotic, save_json, load_json


class CharTokenizer:
    """
    Character-level tokenizer for the Transformer model.
    Handles encoding of text to integer sequences and decoding back to text.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.special_tokens = Config.SPECIAL_TOKENS

        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token

    def fit_on_texts(self, texts):
        """
        Builds vocabulary from a list of texts.
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

    def encode(self, text, add_special_tokens=False):
        """
        Converts a string to a list of token IDs.
        """
        text = str(text)
        ids = []
        if add_special_tokens:
            ids.append(Config.SOS_IDX)

        for char in text:
            ids.append(self.char2idx.get(char, Config.UNK_IDX))

        if add_special_tokens:
            ids.append(Config.EOS_IDX)
        return ids

    def decode(self, ids, remove_special_tokens=True):
        """
        Converts a list of token IDs back to a string.
        """
        chars = []
        for idx in ids:
            # Handle tensor or int
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            if remove_special_tokens and idx in [
                Config.PAD_IDX,
                Config.SOS_IDX,
                Config.EOS_IDX,
                Config.SEP_IDX,
                Config.UNK_IDX,
            ]:
                continue

            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def save(self, path):
        save_json(
            {
                "char2idx": self.char2idx,
                "idx2char": {k: v for k, v in self.idx2char.items()},
            },
            path,
        )

    def load(self, path):
        data = load_json(path)
        self.char2idx = data["char2idx"]
        # JSON keys are always strings, convert back to int for idx2char
        self.idx2char = {int(k): v for k, v in data["idx2char"].items()}

    def __len__(self):
        return len(self.char2idx)


class TransformerDataset(Dataset):
    """
    PyTorch Dataset for the Tier 2 Transformer.
    Input: <prev_word> <SEP> <target_token> <SEP> <next_word>
    Target: <after_token>
    """

    def __init__(self, df, tokenizer, max_len=Config.MAX_SEQ_LEN, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct source string
        # Ensure values are strings (handle potential NaNs converted to string "nan" or floats)
        prev_w = str(row["prev"])
        curr_w = str(row["before"])
        next_w = str(row["next"])

        # Format: <prev> <SEP> <target> <SEP> <next>
        # We process this as a single sequence of characters, inserting the SEP token explicitly
        # However, CharTokenizer encodes characters. We need to insert the SEP_ID manually between the encoded words.

        prev_ids = self.tokenizer.encode(prev_w, add_special_tokens=False)
        curr_ids = self.tokenizer.encode(curr_w, add_special_tokens=False)
        next_ids = self.tokenizer.encode(next_w, add_special_tokens=False)

        sep = [Config.SEP_IDX]

        input_ids = prev_ids + sep + curr_ids + sep + next_ids

        # Truncate if necessary (keeping the center/target is important, but simple truncation is usually fine for chars)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]

        # Pad
        pad_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [Config.PAD_IDX] * pad_len

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "original_text": curr_w,
            "id": row.get("id_str", ""),  # Helper for submission reconstruction
        }

        if not self.is_test:
            target_text = str(row["after"])
            # Target needs SOS and EOS for autoregressive training
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=True)

            # Pad target
            tgt_pad_len = Config.MAX_TARGET_LEN - len(target_ids)
            if tgt_pad_len < 0:
                target_ids = target_ids[: Config.MAX_TARGET_LEN - 1] + [Config.EOS_IDX]
                tgt_pad_len = 0

            target_ids = target_ids + [Config.PAD_IDX] * tgt_pad_len
            result["labels"] = torch.tensor(target_ids, dtype=torch.long)

        return result


def _add_context_and_filter(df, is_train=True, cache_path=None, load_cached_data=True):
    """
    Helper to add context (prev/next words) and filter for semiotic tokens.
    Uses caching to speed up repeated runs.
    """
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached processed data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Processing data: Adding context and filtering...")
    # Ensure sorting by sentence_id and token_id
    # Assuming data is already roughly sorted, but let's be safe
    if "sentence_id" in df.columns and "token_id" in df.columns:
        df = df.sort_values(["sentence_id", "token_id"])

    # Vectorized context extraction
    # Shift 'before' column
    df["prev"] = df["before"].shift(1).fillna(Config.SOS_TOKEN)
    df["next"] = df["before"].shift(-1).fillna(Config.EOS_TOKEN)

    # Handle sentence boundaries
    # If sentence_id changes, prev of current is SOS, next of previous is EOS
    s_ids = df["sentence_id"]

    # Start of sentence: current sentence_id != prev sentence_id
    is_start = s_ids != s_ids.shift(1)
    # End of sentence: current sentence_id != next sentence_id
    is_end = s_ids != s_ids.shift(-1)

    df.loc[is_start, "prev"] = Config.SOS_TOKEN
    df.loc[is_end, "next"] = Config.EOS_TOKEN

    # Construct ID string for submission mapping if not present
    if "id" not in df.columns:
        df["id_str"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)
    else:
        df["id_str"] = df["id"]

    # Filter for semiotic tokens (containing digits) if training/val
    # For test set, we might keep all or filter later.
    # The requirement says Tier 2 is for semiotic.
    # We will filter here to create the Transformer dataset.
    if is_train:
        # Filter
        mask = df["before"].astype(str).apply(is_semiotic)
        df_filtered = df[mask].copy()
        print(f"Filtered {len(df)} -> {len(df_filtered)} semiotic tokens.")
        df = df_filtered

    # Cache result
    if cache_path:
        print(f"Saving processed data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def get_hfbb_data():
    """
    Loads the raw training and validation data for the HFBB (Tier 1) component.
    No filtering is applied.
    """
    print("Loading raw HFBB data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    return train_df, val_df


def build_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for the Transformer model.
    1. Loads metadata.
    2. Adds context (prev/next).
    3. Filters for semiotic tokens (containing digits).
    4. Builds/Loads Vocabulary.
    5. Returns DataLoaders.
    """
    Config.setup_directories()

    # Paths for cached files
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")

    # Load and process data
    print("Preparing Training Data...")
    df_train_raw = pd.read_csv(Config.TRAIN_CSV)
    df_train = _add_context_and_filter(
        df_train_raw,
        is_train=True,
        cache_path=train_cache,
        load_cached_data=load_cached_data,
    )

    print("Preparing Validation Data...")
    df_val_raw = pd.read_csv(Config.VAL_CSV)
    df_val = _add_context_and_filter(
        df_val_raw,
        is_train=True,
        cache_path=val_cache,
        load_cached_data=load_cached_data,
    )

    # Build or Load Vocabulary
    tokenizer = CharTokenizer()
    if load_cached_data and os.path.exists(Config.VOCAB_PATH):
        print(f"Loading vocabulary from {Config.VOCAB_PATH}...")
        tokenizer.load(Config.VOCAB_PATH)
    else:
        print("Building vocabulary from training data...")
        # Collect all relevant text for vocab building
        # We need characters from 'before', 'after', 'prev', 'next'
        texts = []
        texts.extend(df_train["before"].astype(str).tolist())
        texts.extend(df_train["after"].astype(str).tolist())
        texts.extend(df_train["prev"].astype(str).tolist())
        texts.extend(df_train["next"].astype(str).tolist())

        tokenizer.fit_on_texts(texts)
        tokenizer.save(Config.VOCAB_PATH)
        print(f"Vocabulary saved to {Config.VOCAB_PATH}. Size: {len(tokenizer)}")

    # Create Datasets
    train_dataset = TransformerDataset(df_train, tokenizer)
    val_dataset = TransformerDataset(df_val, tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer


def build_test_dataloader(tokenizer, load_cached_data=True):
    """
    Prepares DataLoader for the Test set.
    Filters for semiotic tokens to run through Tier 2.
    """
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    print("Preparing Test Data...")
    if load_cached_data and os.path.exists(test_cache):
        df_test = pd.read_parquet(test_cache)
    else:
        df_test_raw = pd.read_csv(Config.TEST_CSV)
        # We use is_train=True logic here effectively to filter semiotic,
        # but we need to be careful. The inference pipeline needs to know WHICH tokens were predicted.
        # So we filter, and the dataset returns the ID.
        df_test = _add_context_and_filter(
            df_test_raw,
            is_train=True,
            cache_path=test_cache,
            load_cached_data=load_cached_data,
        )

    test_dataset = TransformerDataset(df_test, tokenizer, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
