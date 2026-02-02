import os
import pandas as pd
import torch
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List, Dict, Optional, Tuple, Union
from library.config import Config

# ==========================================
# Data Processing & Caching Functions
# ==========================================


def process_router_data(
    split: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads, groups, and filters data for the Router model.
    Groups tokens into sentences. Performs strategic sampling on training data.

    Args:
        split: 'train', 'val', or 'test'.
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Grouped dataframe with columns ['sentence_id', 'tokens', 'labels', 'token_ids'].
                      (Labels are missing for test split).
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"router_{split}_grouped.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached router data for {split} from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing router data for {split} from scratch...")

    # 2. Determine File Path
    if split == "train":
        file_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        file_path = Config.VAL_DATA_PATH
    elif split == "test":
        file_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 3. Load Raw Data
    # keep_default_na=False is critical for tokens like "null" or "nan"
    df = pd.read_csv(file_path, keep_default_na=False, dtype={"sentence_id": int})

    # Debug Sampling
    if Config.DEBUG_SAMPLE_SIZE is not None:
        # We sample by sentence_id to keep sentences intact, but for raw load speed we just slice
        # then filter complete sentences.
        # Better: just take head, it's debug.
        df = df.head(Config.DEBUG_SAMPLE_SIZE * 20)  # Approx 20 tokens per sentence

    # 4. Group by Sentence
    # We need to aggregate tokens into lists.
    # For train/val, we also aggregate classes.
    # For test, we aggregate ids (for submission mapping).

    if split == "test":
        # Grouping for test
        grouped = (
            df.groupby("sentence_id", sort=False)
            .agg({"before": list, "id": list})
            .reset_index()
        )
        grouped.rename(columns={"before": "tokens", "id": "token_ids"}, inplace=True)
    else:
        # Grouping for train/val
        grouped = (
            df.groupby("sentence_id", sort=False)
            .agg({"before": list, "class": list, "id": list})
            .reset_index()
        )
        grouped.rename(
            columns={"before": "tokens", "class": "labels", "id": "token_ids"},
            inplace=True,
        )

        # 5. Strategic Sampling (Train Only)
        if split == "train":
            print(f"Applying strategic sampling to {len(grouped)} sentences...")

            def sampling_filter(row):
                labels = row["labels"]
                tokens = row["tokens"]

                # Check for non-PLAIN classes
                has_non_plain = any(l != "PLAIN" for l in labels)
                if has_non_plain:
                    return True

                # Hard Negative Mining for PLAIN sentences
                # Keep if contains digits or uppercase (potential confusion points)
                # We join to search faster than iterating
                text_blob = "".join(tokens)
                if re.search(r"[A-Z0-9]", text_blob):
                    return True

                # Discard trivial lowercase plain sentences
                return False

            # Apply filter
            initial_count = len(grouped)
            grouped = grouped[grouped.apply(sampling_filter, axis=1)].reset_index(
                drop=True
            )
            print(
                f"Reduced training set from {initial_count} to {len(grouped)} sentences."
            )

    # 6. Save Cache
    # Ensure lists are stored correctly in parquet (pyarrow handles lists usually)
    print(f"Saving processed router data to {cache_path}...")
    grouped.to_parquet(cache_path, index=False)

    return grouped


def process_generator_data(
    split: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads and filters data for the Generator model.
    Keeps only tokens belonging to UNSTRUCTURED_CLASSES.

    Args:
        split: 'train' or 'val'. (Generator is not directly run on test csv in this way).
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Filtered dataframe with columns ['class', 'before', 'after'].
    """
    if split == "test":
        raise ValueError(
            "Generator data processing is not applicable for test split directly (inference is dynamic)."
        )

    cache_path = os.path.join(Config.CACHE_DIR, f"generator_{split}_filtered.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached generator data for {split} from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing generator data for {split} from scratch...")

    if split == "train":
        file_path = Config.TRAIN_DATA_PATH
    else:
        file_path = Config.VAL_DATA_PATH

    df = pd.read_csv(file_path, keep_default_na=False)

    if Config.DEBUG_SAMPLE_SIZE is not None:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Filter for unstructured classes
    # We use the set defined in Config
    unstructured_set = set(Config.UNSTRUCTURED_CLASSES)

    filtered = df[df["class"].isin(unstructured_set)].copy()

    # We only need class, before, after
    filtered = filtered[["class", "before", "after"]].reset_index(drop=True)

    print(f"Filtered {split} data: {len(filtered)} unstructured tokens found.")

    print(f"Saving processed generator data to {cache_path}...")
    filtered.to_parquet(cache_path, index=False)

    return filtered


# ==========================================
# Dataset Classes
# ==========================================


class RouterDataset(Dataset):
    """
    Dataset for token classification (Router).
    Aligns word-level labels to subword tokens.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 128,
        is_test: bool = False,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-compute label map for speed
        self.class2id = Config.CLASS2ID

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row["tokens"]  # List of strings

        # Tokenize
        # is_split_into_words=True indicates that input is already split by whitespace/punctuation
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding=False,  # We pad in collate_fn
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            labels = row["labels"]  # List of strings
            word_ids = encoding.word_ids()

            # Align labels
            # -100 is the ignore index for PyTorch CrossEntropyLoss
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (CLS, SEP)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a new word -> assign label
                    # Ensure we don't go out of bounds if truncation happened
                    if word_idx < len(labels):
                        label_str = labels[word_idx]
                        label_ids.append(
                            self.class2id.get(label_str, 0)
                        )  # Default to PLAIN (0) if issue
                    else:
                        label_ids.append(-100)
                else:
                    # Subsequent subwords -> ignore
                    label_ids.append(-100)
                previous_word_idx = word_idx

            result["labels"] = label_ids
        else:
            # For test set, we might want to pass token_ids to reconstruct submission
            # But usually, we just iterate sequentially.
            # We'll return the sentence_id or token_ids for tracking if needed,
            # but standard DataLoader flow usually relies on order.
            pass

        return result


class GeneratorDataset(Dataset):
    """
    Dataset for Seq2Seq normalization (Generator).
    Input: "[CLASS] raw_token"
    Target: "normalized text"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_input_len: int = 128,
        max_target_len: int = 128,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cls_label = row["class"]
        raw_text = row["before"]
        target_text = row["after"]

        # Format input: "[CLASS] raw_text"
        # Note: ByT5 is robust, we can just use a simple separator or format
        input_text = f"[{cls_label}] {raw_text}"

        # Tokenize Input
        input_enc = self.tokenizer(
            input_text, truncation=True, max_length=self.max_input_len, padding=False
        )

        # Tokenize Target
        with self.tokenizer.as_target_tokenizer():
            target_enc = self.tokenizer(
                target_text,
                truncation=True,
                max_length=self.max_target_len,
                padding=False,
            )

        return {
            "input_ids": input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


# ==========================================
# Collate Functions
# ==========================================


def router_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Dynamic padding for Router batches.
    """
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]

    # Pad inputs
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=1
    )  # 1 is usually pad for Roberta, but check tokenizer. Using 1 is safe for RoBERTa (pad_token_id).
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    result = {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded}

    if "labels" in batch[0]:
        labels = [torch.tensor(item["labels"]) for item in batch]
        # Pad labels with -100 (ignore index)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        result["labels"] = labels_padded

    return result


def generator_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Dynamic padding for Generator batches.
    """
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    # Pad
    # ByT5 pad_token_id is 0
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )  # -100 for loss calculation

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }


# ==========================================
# DataLoader Builders
# ==========================================


def get_router_dataloader(
    split: str = "train", load_cached_data: bool = True
) -> DataLoader:
    """
    Creates a DataLoader for the Router model.
    """
    df = process_router_data(split=split, load_cached_data=load_cached_data)

    tokenizer = AutoTokenizer.from_pretrained(
        Config.ROUTER_MODEL_NAME, add_prefix_space=True
    )

    dataset = RouterDataset(
        df=df,
        tokenizer=tokenizer,
        max_len=Config.ROUTER_MAX_LEN,
        is_test=(split == "test"),
    )

    batch_size = (
        Config.ROUTER_TRAIN_BATCH_SIZE
        if split == "train"
        else Config.ROUTER_VAL_BATCH_SIZE
    )
    shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        collate_fn=router_collate_fn,
        pin_memory=True,
    )


def get_generator_dataloader(
    split: str = "train", load_cached_data: bool = True
) -> DataLoader:
    """
    Creates a DataLoader for the Generator model.
    """
    df = process_generator_data(split=split, load_cached_data=load_cached_data)

    tokenizer = AutoTokenizer.from_pretrained(Config.GENERATOR_MODEL_NAME)

    dataset = GeneratorDataset(
        df=df,
        tokenizer=tokenizer,
        max_input_len=Config.GEN_MAX_INPUT_LEN,
        max_target_len=Config.GEN_MAX_TARGET_LEN,
    )

    batch_size = (
        Config.GEN_TRAIN_BATCH_SIZE if split == "train" else Config.GEN_VAL_BATCH_SIZE
    )
    shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        collate_fn=generator_collate_fn,
        pin_memory=True,
    )
