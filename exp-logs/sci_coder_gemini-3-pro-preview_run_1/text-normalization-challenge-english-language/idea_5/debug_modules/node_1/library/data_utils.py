import os
import pandas as pd
import numpy as np
from collections import Counter
import torch
from library.config import Config


class Vocabulary:
    """
    Handles mapping between tokens (strings) and indices (integers).
    Supports saving/loading to Parquet.
    """

    def __init__(self, name, specials=None):
        self.name = name
        self.stoi = {}
        self.itos = {}
        self.specials = specials if specials else []

        # Initialize with specials
        for i, s in enumerate(self.specials):
            self.stoi[s] = i
            self.itos[i] = s

    def __len__(self):
        return len(self.stoi)

    def add_token(self, token):
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token

    def lookup_indices(self, tokens, unk_token=None):
        unk_idx = self.stoi.get(unk_token) if unk_token else None
        return [self.stoi.get(t, unk_idx) for t in tokens]

    def lookup_tokens(self, indices):
        return [self.itos.get(i, Config.UNK_TOKEN) for i in indices]

    def save(self, path):
        """Saves vocabulary to a parquet file."""
        data = []
        for token, idx in self.stoi.items():
            data.append({"token": token, "index": idx})
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path):
        """Loads vocabulary from a parquet file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        df = pd.read_parquet(path)
        self.stoi = dict(zip(df["token"], df["index"]))
        self.itos = dict(zip(df["index"], df["token"]))
        # Ensure specials are tracked if needed, though stoi/itos is the source of truth

    @classmethod
    def from_counter(cls, name, counter, specials, max_size=None, min_freq=1):
        vocab = cls(name, specials)

        # Sort by frequency then alphabetically
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if max_size and len(vocab) >= max_size:
                break
            vocab.add_token(token)
        return vocab


def build_vocabularies(df_train, load_cached_data=True):
    """
    Builds word, character, and class vocabularies from training data.
    Implements caching logic.
    """
    # Define paths
    word_path = Config.VOCAB_WORDS_PATH
    char_path = Config.VOCAB_CHARS_PATH
    class_path = Config.VOCAB_CLASSES_PATH

    # Check cache
    if (
        load_cached_data
        and os.path.exists(word_path)
        and os.path.exists(char_path)
        and os.path.exists(class_path)
    ):

        vocab_words = Vocabulary("words")
        vocab_words.load(word_path)

        vocab_chars = Vocabulary("chars")
        vocab_chars.load(char_path)

        vocab_classes = Vocabulary("classes")
        vocab_classes.load(class_path)

        return vocab_words, vocab_chars, vocab_classes

    # Compute from scratch
    print("Building vocabularies from scratch...")

    # 1. Word Vocabulary
    # Convert to string to ensure no type issues
    tokens = df_train["before"].astype(str).tolist()
    word_counter = Counter(tokens)

    vocab_words = Vocabulary.from_counter(
        "words",
        word_counter,
        specials=[Config.PAD_TOKEN, Config.UNK_TOKEN],
        max_size=Config.MAX_VOCAB_SIZE,
        min_freq=Config.MIN_FREQ,
    )

    # 2. Character Vocabulary
    # Get all unique characters
    char_counter = Counter()
    for t in tokens:
        char_counter.update(t)

    vocab_chars = Vocabulary.from_counter(
        "chars",
        char_counter,
        specials=[
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
        ],
        # No max size for chars usually, but can limit if needed.
        # Usually char vocab is small (<200).
    )

    # 3. Class Vocabulary
    classes = df_train["class"].astype(str).tolist()
    class_counter = Counter(classes)

    vocab_classes = Vocabulary.from_counter(
        "classes",
        class_counter,
        specials=[Config.PAD_TOKEN],  # Padding for class labels in batching
        min_freq=1,
    )

    # Save to cache
    print(f"Saving vocabularies to {Config.WORKING_DIR}...")
    vocab_words.save(word_path)
    vocab_chars.save(char_path)
    vocab_classes.save(class_path)

    return vocab_words, vocab_chars, vocab_classes


def build_knowledge_base(df_train, load_cached_data=True):
    """
    Constructs a deterministic dictionary mapping (before, class) -> after.
    """
    kb_path = Config.KNOWLEDGE_BASE_PATH

    if load_cached_data and os.path.exists(kb_path):
        df_kb = pd.read_parquet(kb_path)
        # Convert dataframe back to dictionary map
        # We use a tuple key (before, class)
        kb = {}
        # Iterating dataframe rows is slow, zip is faster
        for b, c, a in zip(df_kb["before"], df_kb["class"], df_kb["after"]):
            kb[(b, c)] = a
        return kb

    print("Building Knowledge Base from scratch...")

    # We want the most frequent 'after' for each (before, class) pair
    # Group by before+class and find mode of after
    # Since we need to be deterministic, we can take the most common one.

    # Create a composite key for grouping to avoid multi-index complexity in simple aggregation
    # But standard groupby is fine.

    # Filter valid data
    df = df_train[["before", "class", "after"]].astype(str).copy()

    # Count occurrences of each mapping
    mapping_counts = (
        df.groupby(["before", "class", "after"]).size().reset_index(name="count")
    )

    # Sort by count descending so the first one is the most frequent
    mapping_counts = mapping_counts.sort_values(
        ["before", "class", "count"], ascending=[True, True, False]
    )

    # Drop duplicates keeping the first (most frequent)
    best_mappings = mapping_counts.drop_duplicates(
        subset=["before", "class"], keep="first"
    )

    # Create dictionary
    kb = {}
    kb_list = []
    for _, row in best_mappings.iterrows():
        b, c, a = row["before"], row["class"], row["after"]
        kb[(b, c)] = a
        kb_list.append({"before": b, "class": c, "after": a})

    # Save to cache
    print(f"Saving Knowledge Base to {kb_path}...")
    df_kb_save = pd.DataFrame(kb_list)
    df_kb_save.to_parquet(kb_path, index=False)

    return kb


def load_and_group_data(split, load_cached_data=True):
    """
    Loads data from metadata CSVs and groups by sentence_id.
    Returns a DataFrame where each row is a sentence with lists of tokens.

    Args:
        split (str): 'train', 'val', or 'test'
        load_cached_data (bool): Whether to use cached parquet files.
    """
    if split == "train":
        input_path = Config.TRAIN_DATA_PATH
        output_path = Config.TRAIN_GROUPED_PATH
    elif split == "val":
        input_path = Config.VAL_DATA_PATH
        output_path = Config.VAL_GROUPED_PATH
    elif split == "test":
        input_path = Config.TEST_DATA_PATH
        output_path = Config.TEST_GROUPED_PATH
    else:
        raise ValueError("Invalid split name")

    # Check cache
    if load_cached_data and os.path.exists(output_path):
        # Reading parquet with list columns works out of the box in pandas/pyarrow
        return pd.read_parquet(output_path)

    print(f"Processing {split} data from {input_path}...")

    # Load raw CSV
    # keep_default_na=False is crucial to not lose tokens like "null" or "NaN"
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)

    # Ensure ID columns are numeric for sorting
    df["sentence_id"] = pd.to_numeric(df["sentence_id"])
    df["token_id"] = pd.to_numeric(df["token_id"])

    # Sort by sentence_id then token_id to ensure order
    df = df.sort_values(["sentence_id", "token_id"])

    # Define aggregation dict
    agg_dict = {"token_id": list, "before": list, "id": list}

    # Add target columns if they exist
    if "class" in df.columns:
        agg_dict["class"] = list
    if "after" in df.columns:
        agg_dict["after"] = list

    # Group
    df_grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

    # Save to cache
    print(f"Saving grouped {split} data to {output_path}...")
    df_grouped.to_parquet(output_path, index=False)

    return df_grouped


def load_raw_data(split):
    """
    Helper to load raw CSV without grouping, useful for initial vocab building.
    """
    if split == "train":
        path = Config.TRAIN_DATA_PATH
    elif split == "val":
        path = Config.VAL_DATA_PATH
    elif split == "test":
        path = Config.TEST_DATA_PATH
    else:
        raise ValueError("Invalid split")

    return pd.read_csv(path, dtype=str, keep_default_na=False)
