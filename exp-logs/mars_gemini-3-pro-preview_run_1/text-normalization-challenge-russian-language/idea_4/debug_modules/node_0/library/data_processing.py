import os
import json
import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_or_compute, ensure_dir


class CharTokenizer:
    """
    Character-level tokenizer that handles special tokens as atomic units.
    """

    def __init__(self, config):
        self.config = config
        self.char_to_id = {}
        self.id_to_char = {}

        # Initialize with special tokens from config
        # We assign IDs 0 to N-1 for special tokens
        for idx, token in enumerate(config.special_tokens):
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def fit_on_texts(self, texts):
        """
        Updates vocabulary based on the provided list of texts.
        """
        unique_chars = set()
        for text in texts:
            if not isinstance(text, str):
                continue
            # Remove special tokens from consideration for raw chars if they appear in text
            # (Though usually special tokens are added structurally, not naturally in text)
            # We just collect all characters
            unique_chars.update(text)

        # Filter out characters that are already special tokens (unlikely but safe)
        unique_chars = sorted(list(unique_chars - set(self.config.special_tokens)))

        start_idx = len(self.char_to_id)
        for idx, char in enumerate(unique_chars):
            self.char_to_id[char] = start_idx + idx
            self.id_to_char[start_idx + idx] = char

    def save(self, path):
        """Saves the vocabulary to a JSON file."""
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.char_to_id = json.load(f)
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}

    def encode(self, text, add_special_tokens=False):
        """
        Converts a string to a list of IDs.
        Handles special tokens by splitting the string using regex.
        """
        if not isinstance(text, str):
            return []

        # Create a regex pattern to split by special tokens
        # Escape special tokens to handle regex special chars like '.' or '?'
        escaped_specials = [re.escape(t) for t in self.config.special_tokens]
        pattern = f"({'|'.join(escaped_specials)})"

        parts = re.split(pattern, text)
        ids = []

        if add_special_tokens:
            ids.append(self.char_to_id[self.config.bos_token])

        unk_id = self.char_to_id.get(self.config.unk_token)

        for part in parts:
            if not part:
                continue
            if part in self.char_to_id:
                # It's a special token (or a single char that happens to be in vocab)
                ids.append(self.char_to_id[part])
            else:
                # It's a sequence of characters
                for char in part:
                    ids.append(self.char_to_id.get(char, unk_id))

        if add_special_tokens:
            ids.append(self.char_to_id[self.config.eos_token])

        return ids

    def decode(self, ids, remove_special_tokens=True):
        """
        Converts a list of IDs back to a string.
        """
        chars = []
        for i in ids:
            token = self.id_to_char.get(i, "")
            if remove_special_tokens and token in self.config.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)


def _generate_sequences(config, split):
    """
    Internal function to generate the processed dataframe.
    This is passed to get_or_compute for caching.
    """
    # 1. Load Metadata
    if split == "train":
        filepath = config.train_file
    elif split == "val":
        filepath = config.val_file
    elif split == "test":
        filepath = config.test_file
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load data with object dtype to preserve tokens exactly
    df = pd.read_csv(
        filepath, dtype={"before": object, "after": object, "class": object}
    )

    # Fill NaNs
    df["before"] = df["before"].fillna("")
    if "after" in df.columns:
        df["after"] = df["after"].fillna("")
    if "class" in df.columns:
        df["class"] = df["class"].fillna("UNKNOWN")

    # 2. Reconstruct Sentences
    # Group tokens by sentence_id to form the full sentence context
    # We use a dictionary for fast lookups: sentence_id -> list of tokens
    # Note: df is assumed to be sorted by sentence_id, token_id, but we enforce it just in case
    # Sorting is expensive, so we assume metadata is well-formed or we rely on groupby order

    # Optimization: Use groupby on the 'before' column
    # This creates a Series where index is sentence_id and value is list of tokens
    sentences_map = df.groupby("sentence_id")["before"].apply(list).to_dict()

    # 3. Filter Rows (for Train/Val)
    if split in ["train", "val"]:
        # Identify "Hard" tokens: Not PLAIN/PUNCT OR contains digits
        is_plain_punct = df["class"].isin(["PLAIN", "PUNCT"])
        has_digits = df["before"].str.contains(r"\d", regex=True)
        is_hard = (~is_plain_punct) | has_digits

        # Identify "Easy" tokens to sample
        # We want to include a ratio of PLAIN/PUNCT to learn grammar
        # We use a deterministic random choice based on index or hash if we wanted strict determinism,
        # but np.random with fixed seed in Config is sufficient.
        # However, to be vectorized:
        np.random.seed(config.seed)
        random_vals = np.random.rand(len(df))
        is_selected_easy = is_plain_punct & (random_vals < config.plain_inclusion_ratio)

        # Combine masks
        keep_mask = is_hard | is_selected_easy
        df_filtered = df[keep_mask].copy()
    else:
        # For test, we process everything (or let the router decide later).
        # To be safe and allow the model to predict anything, we keep all.
        df_filtered = df.copy()

    # 4. Format Inputs with Context
    # We need to construct: "... prev <tgt> token </tgt> next ..."
    # Iterating row-by-row is slow in pure Python, but necessary for the context insertion.
    # We can optimize by iterating only the filtered dataframe.

    input_texts = []
    target_texts = []
    ids = []

    # Pre-fetch config tokens to avoid dot lookup in loop
    tgt_start = config.tgt_start_token
    tgt_end = config.tgt_end_token

    # Iterate as tuples for speed
    # rows: (sentence_id, token_id, before, after)
    # Note: 'after' might not exist in test
    has_target = "after" in df_filtered.columns

    # Convert to records for iteration
    records = df_filtered.to_dict("records")

    for row in records:
        s_id = row["sentence_id"]
        t_id = row["token_id"]

        # Get full sentence tokens
        # Copy list to avoid modifying the reference for other tokens in same sentence
        sent_tokens = sentences_map[s_id].copy()

        # Safety check for index
        if t_id >= len(sent_tokens):
            continue

        # Insert tags
        # We insert placeholders or the tokens themselves.
        # Ideally: "word" -> "<tgt> word </tgt>"
        # We replace the token at t_id with the tagged version
        original_token = sent_tokens[t_id]
        tagged_token = f"{tgt_start} {original_token} {tgt_end}"
        sent_tokens[t_id] = tagged_token

        # Join sentence
        input_str = " ".join(sent_tokens)
        input_texts.append(input_str)

        # Handle Target
        if has_target:
            target_texts.append(row["after"])
        else:
            target_texts.append("")

        # Keep track of ID for submission
        ids.append(f"{s_id}_{t_id}")

    # Create result dataframe
    result_df = pd.DataFrame(
        {"id": ids, "input_text": input_texts, "target_text": target_texts}
    )

    return result_df


def prepare_neural_dataset(
    config, split="train", tokenizer=None, load_cached_data=True
):
    """
    Main function to prepare data.
    1. Generates or loads processed dataframe (Parquet).
    2. Fits or loads tokenizer.

    Args:
        config: Config object.
        split: 'train', 'val', or 'test'.
        tokenizer: Optional existing tokenizer instance.
        load_cached_data: Whether to use caching.

    Returns:
        df: Processed dataframe.
        tokenizer: The fitted/loaded tokenizer.
    """
    # Determine cache path based on split and config hash
    if split == "train":
        cache_path = config.train_seq_path
    elif split == "val":
        cache_path = config.val_seq_path
    elif split == "test":
        cache_path = config.test_seq_path
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Get Dataframe (Cached or Computed)
    # We pass the function _generate_sequences and its kwargs
    df = get_or_compute(
        cache_path,
        _generate_sequences,
        load_cached_data=load_cached_data,
        config=config,
        split=split,
    )

    # 2. Handle Tokenizer
    if tokenizer is None:
        tokenizer = CharTokenizer(config)

    tokenizer_path = config.tokenizer_path

    if split == "train":
        # For training, we either load an existing tokenizer or fit a new one
        if load_cached_data and os.path.exists(tokenizer_path):
            tokenizer.load(tokenizer_path)
        else:
            # Fit on the training data
            # We combine inputs and targets to cover all characters
            print("Fitting tokenizer on training data...")
            all_texts = df["input_text"].tolist() + df["target_text"].tolist()
            tokenizer.fit_on_texts(all_texts)
            tokenizer.save(tokenizer_path)
    else:
        # For val/test, we must use the existing tokenizer
        # If it's not provided and not on disk, we can't properly tokenize
        if tokenizer.vocab_size <= len(
            config.special_tokens
        ):  # Check if empty/init only
            if os.path.exists(tokenizer_path):
                tokenizer.load(tokenizer_path)
            else:
                # If we are in test mode and no tokenizer exists, this is critical.
                # However, usually train is run first.
                # We will log a warning or rely on the caller to handle order.
                print(
                    f"Warning: Tokenizer not found at {tokenizer_path} for split {split}. Vocab might be incomplete."
                )

    return df, tokenizer


class TextNormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    """

    def __init__(self, df, tokenizer, config, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode  # 'train', 'val', 'test'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_text = row["input_text"]
        target_text = row["target_text"]

        # Tokenize Input
        # We don't add BOS/EOS to encoder input usually, or just standard
        # For this architecture, let's assume standard encoder input
        src_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

        # Tokenize Target
        if self.mode in ["train", "val"]:
            # For decoder, we need BOS and EOS
            tgt_ids = self.tokenizer.encode(target_text, add_special_tokens=True)
        else:
            tgt_ids = []

        return {
            "id": row["id"],
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "raw_input": input_text,
            "raw_target": target_text,
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences.
    """
    # Extract sequences
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]
    ids = [item["id"] for item in batch]

    # Pad sequences
    # Assuming padding_value is the index of <pad>
    # We need to access the tokenizer or config to know the pad ID.
    # Since collate_fn is usually standalone, we might need to hardcode or pass it.
    # However, CharTokenizer initializes special tokens first.
    # <pad> is usually index 0 if it's first in config.special_tokens.
    # Let's assume Config.pad_token is first.
    pad_id = 0

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=pad_id
    )

    if len(tgt_list[0]) > 0:
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_list, batch_first=True, padding_value=pad_id
        )
    else:
        tgt_padded = None

    return {"id": ids, "src": src_padded, "tgt": tgt_padded}
