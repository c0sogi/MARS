import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed()


class Vocabulary:
    """
    Handles mapping between tokens/classes and integer indices.
    """

    def __init__(self, tokens=None, specials=None):
        self.stoi = {}
        self.itos = {}
        self.specials = specials if specials else []

        if tokens:
            self.build(tokens)

    def build(self, tokens):
        idx = 0
        # Add special tokens first
        for s in self.specials:
            self.stoi[s] = idx
            self.itos[idx] = s
            idx += 1

        # Add regular tokens
        for t in tokens:
            if t not in self.stoi:
                self.stoi[t] = idx
                self.itos[idx] = t
                idx += 1

    def __len__(self):
        return len(self.stoi)

    def lookup_indices(self, tokens):
        unk_idx = self.stoi.get(Config.UNK_TOKEN, 0)
        return [self.stoi.get(t, unk_idx) for t in tokens]

    def lookup_token(self, idx):
        return self.itos.get(idx, Config.UNK_TOKEN)

    def save(self, path):
        data = {"token": list(self.stoi.keys()), "index": list(self.stoi.values())}
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    @classmethod
    def load(cls, path):
        df = pd.read_parquet(path)
        vocab = cls()
        vocab.stoi = dict(zip(df["token"], df["index"]))
        vocab.itos = dict(zip(df["index"], df["token"]))
        return vocab


class TextNormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    Groups tokens by sentence and handles padding/masking.
    """

    def __init__(self, df, vocab_tokens, vocab_classes, max_len, is_test=False):
        self.df = df
        self.vocab_tokens = vocab_tokens
        self.vocab_classes = vocab_classes
        self.max_len = max_len
        self.is_test = is_test
        self.pad_token_id = vocab_tokens.stoi.get(Config.PAD_TOKEN, 0)
        self.pad_class_id = vocab_classes.stoi.get(Config.PAD_TOKEN, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Inputs ---
        raw_tokens = row["before"]
        if isinstance(raw_tokens, np.ndarray):
            raw_tokens = raw_tokens.tolist()

        # Truncate to max_len
        raw_tokens = raw_tokens[: self.max_len]
        seq_len = len(raw_tokens)

        # Convert to IDs
        token_ids = self.vocab_tokens.lookup_indices(raw_tokens)

        # Padding
        pad_len = self.max_len - seq_len
        input_ids = token_ids + [self.pad_token_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        # Submission IDs (needed for final output)
        submission_ids = row["id"]
        if isinstance(submission_ids, np.ndarray):
            submission_ids = submission_ids.tolist()
        submission_ids = submission_ids[: self.max_len]
        # Pad submission ids with empty strings to keep collate happy, though we'll only use valid ones
        submission_ids = submission_ids + [""] * pad_len

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "submission_ids": submission_ids,
            "raw_tokens": raw_tokens + [Config.PAD_TOKEN] * pad_len,  # For lookup
        }

        # --- Targets (Train/Val only) ---
        if not self.is_test:
            raw_classes = row["class"]
            if isinstance(raw_classes, np.ndarray):
                raw_classes = raw_classes.tolist()
            raw_classes = raw_classes[: self.max_len]

            class_ids = self.vocab_classes.lookup_indices(raw_classes)
            class_ids = class_ids + [self.pad_class_id] * pad_len

            item["class_ids"] = torch.tensor(class_ids, dtype=torch.long)

        return item


def load_and_group_data(csv_path, cache_path, load_cached=True, is_test=False):
    """
    Loads raw CSV data, groups by sentence_id, and caches result.
    """
    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached grouped data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing raw data from {csv_path}")
    # Load raw csv
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # Define aggregation strategy
    agg_dict = {"before": list, "id": list}
    if not is_test:
        agg_dict["class"] = list
        agg_dict["after"] = list

    # Ensure correct token order
    df["token_id"] = df["token_id"].astype(int)
    df = df.sort_values(["sentence_id", "token_id"])

    # Group by sentence
    grouped = df.groupby("sentence_id").agg(agg_dict).reset_index()

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    grouped.to_parquet(cache_path, index=False)

    return grouped


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get data loaders, vocabularies, and knowledge base.
    """
    # 1. Load Grouped Data
    train_df = load_and_group_data(
        Config.TRAIN_FILE, Config.TRAIN_GROUPED_PATH, load_cached_data, is_test=False
    )
    val_df = load_and_group_data(
        Config.VAL_FILE, Config.VAL_GROUPED_PATH, load_cached_data, is_test=False
    )
    test_df = load_and_group_data(
        Config.TEST_FILE, Config.TEST_GROUPED_PATH, load_cached_data, is_test=True
    )

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG mode enabled: Using only {Config.DEBUG_SIZE} sentences.")
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)
        # We keep test set intact or small depending on need, usually full for submission check
        # but let's truncate for speed if debugging pipeline
        test_df = test_df.head(Config.DEBUG_SIZE)

    # 2. Build/Load Vocabularies (Train only)
    if (
        load_cached_data
        and os.path.exists(Config.VOCAB_TOKENS_PATH)
        and os.path.exists(Config.VOCAB_CLASSES_PATH)
    ):
        print("Loading vocabularies...")
        vocab_tokens = Vocabulary.load(Config.VOCAB_TOKENS_PATH)
        vocab_classes = Vocabulary.load(Config.VOCAB_CLASSES_PATH)
    else:
        print("Building vocabularies...")
        # Flatten tokens
        all_tokens = [t for sublist in train_df["before"] for t in sublist]
        token_counts = Counter(all_tokens)
        tokens = [t for t, c in token_counts.items() if c >= Config.MIN_FREQ]

        vocab_tokens = Vocabulary(
            tokens=tokens, specials=[Config.PAD_TOKEN, Config.UNK_TOKEN]
        )
        vocab_tokens.save(Config.VOCAB_TOKENS_PATH)

        # Classes
        all_classes = [c for sublist in train_df["class"] for c in sublist]
        unique_classes = sorted(list(set(all_classes)))
        vocab_classes = Vocabulary(
            tokens=unique_classes, specials=[Config.PAD_TOKEN, Config.UNK_TOKEN]
        )
        vocab_classes.save(Config.VOCAB_CLASSES_PATH)

    # 3. Build/Load Knowledge Base
    # Mapping: (token, class) -> normalized_text
    if load_cached_data and os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        print("Loading Knowledge Base...")
        kb_df = pd.read_parquet(Config.KNOWLEDGE_BASE_PATH)
        knowledge_base = dict(
            zip(zip(kb_df["token"], kb_df["class_name"]), kb_df["normalized"])
        )
    else:
        print("Building Knowledge Base...")
        # Explode train_df to get token-level rows
        exploded = train_df[["before", "class", "after"]].explode(
            ["before", "class", "after"]
        )

        # Count occurrences of (token, class, normalized)
        counts = (
            exploded.groupby(["before", "class", "after"])
            .size()
            .reset_index(name="count")
        )

        # Sort by count descending to pick most frequent
        counts = counts.sort_values(
            ["before", "class", "count"], ascending=[True, True, False]
        )

        # Drop duplicates to keep top 1
        kb_best = counts.drop_duplicates(subset=["before", "class"])

        # Save
        kb_save_df = kb_best[["before", "class", "after"]].rename(
            columns={"before": "token", "class": "class_name", "after": "normalized"}
        )
        kb_save_df.to_parquet(Config.KNOWLEDGE_BASE_PATH, index=False)

        knowledge_base = dict(
            zip(
                zip(kb_save_df["token"], kb_save_df["class_name"]),
                kb_save_df["normalized"],
            )
        )

    # 4. Create Datasets
    train_dataset = TextNormalizationDataset(
        train_df, vocab_tokens, vocab_classes, Config.MAX_LEN, is_test=False
    )
    val_dataset = TextNormalizationDataset(
        val_df, vocab_tokens, vocab_classes, Config.MAX_LEN, is_test=False
    )
    test_dataset = TextNormalizationDataset(
        test_df, vocab_tokens, vocab_classes, Config.MAX_LEN, is_test=True
    )

    # 5. Create DataLoaders
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        vocab_tokens,
        vocab_classes,
        knowledge_base,
    )
