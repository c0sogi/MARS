import os
import pandas as pd
import numpy as np
import torch
import random
import collections
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Wraps the processed DataFrame and converts rows to tensors.
    """

    def __init__(self, data_df, mode="train"):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing processed features.
            mode (str): 'train' or 'val'/'test'. Determines which columns to return.
        """
        self.data = data_df
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Basic inputs
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        if "token_type_ids" in row and row["token_type_ids"] is not None:
            if hasattr(row["token_type_ids"], "__len__"):
                item["token_type_ids"] = torch.tensor(
                    row["token_type_ids"], dtype=torch.long
                )

        # Add labels for training
        if self.mode == "train":
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)

        # For validation/test, we don't return metadata here as it's handled
        # via the dataframe in the post-processing step.

        return item


def get_tokenizer():
    """
    Initializes and returns the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)


def prepare_train_features(examples_df, tokenizer):
    """
    Preprocesses training data: tokenization, sliding window, and label mapping.
    """
    # Convert DataFrame to dict of lists for tokenizer compatibility
    examples = examples_df.to_dict(orient="list")

    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Extract mappings
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialize dictionary for filtered features
    filtered_features = collections.defaultdict(list)

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Get sequence IDs to distinguish question (0) from context (1)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Retrieve original answer
        sample_index = sample_mapping[i]
        answer_text = examples["answer_text"][sample_index]
        start_char = examples["answer_start"][sample_index]
        end_char = start_char + len(answer_text)

        # Find the start and end of the context in the current window
        # sequence_ids contains None for special tokens, 0 for question, 1 for context
        seq_ids_safe = [x if x is not None else -1 for x in sequence_ids]

        try:
            token_start_index = seq_ids_safe.index(1)
            # Find last index of 1
            token_end_index = len(seq_ids_safe) - 1 - seq_ids_safe[::-1].index(1)
        except ValueError:
            # Context not found in this window (rare case)
            start_pos = cls_index
            end_pos = cls_index
        else:
            # Check if the answer is fully contained in this window
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                start_pos = cls_index
                end_pos = cls_index
            else:
                # Map character positions to token indices
                idx_start = token_start_index
                while (
                    idx_start <= token_end_index and offsets[idx_start][1] <= start_char
                ):
                    idx_start += 1

                idx_end = token_end_index
                while idx_end >= token_start_index and offsets[idx_end][0] >= end_char:
                    idx_end -= 1

                start_pos = idx_start
                end_pos = idx_end

        # Cite solution_lesson_node_00023: Downsample negative windows
        if start_pos == cls_index and end_pos == cls_index:
            if random.random() > Config.NEGATIVE_SAMPLING_PROB:
                continue

        # Add to filtered features
        filtered_features["input_ids"].append(input_ids)
        filtered_features["attention_mask"].append(
            tokenized_examples["attention_mask"][i]
        )
        if "token_type_ids" in tokenized_examples:
            filtered_features["token_type_ids"].append(
                tokenized_examples["token_type_ids"][i]
            )

        filtered_features["start_positions"].append(start_pos)
        filtered_features["end_positions"].append(end_pos)

    return pd.DataFrame(filtered_features)


def prepare_validation_features(examples_df, tokenizer):
    """
    Preprocesses validation/test data: tokenization and sliding window.
    Preserves mappings for post-processing.
    """
    examples = examples_df.to_dict(orient="list")

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We need to construct a dict to convert to DataFrame, handling complex types
    data = dict(tokenized_examples)
    data["example_id"] = []
    data["sequence_ids"] = []

    # Sanitize offset_mapping (convert tuples to lists for Parquet compatibility)
    # offset_mapping is a list of lists of tuples.
    if "offset_mapping" in data:
        data["offset_mapping"] = [
            [list(span) for span in seq] for seq in data["offset_mapping"]
        ]

    for i in range(len(data["input_ids"])):
        sample_index = sample_mapping[i]
        data["example_id"].append(examples["id"][sample_index])

        # Handle sequence_ids (replace None with -1 for storage)
        seq_ids = tokenized_examples.sequence_ids(i)
        data["sequence_ids"].append([s if s is not None else -1 for s in seq_ids])

    return pd.DataFrame(data)


def load_data(split, tokenizer, load_cached_data=True, debug=False):
    """
    Loads, processes, and caches data for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.
        tokenizer: The tokenizer instance.
        load_cached_data (bool): Whether to try loading from cache.
        debug (bool): If True, limits data size for debugging.

    Returns:
        pd.DataFrame: Processed features.
    """
    set_seed(Config.SEED)

    cache_filename = f"cached_{split}_features.parquet"
    if debug:
        cache_filename = f"cached_{split}_debug_features.parquet"

    cache_path = os.path.join(Config.OUTPUT_DIR, cache_filename)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    print(f"Processing {split} data...")

    # 2. Determine input path
    if split == "train":
        input_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        input_path = Config.VAL_DATA_PATH
    elif split == "test":
        input_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 3. Load raw data
    raw_df = pd.read_csv(input_path)

    # Ensure text columns are strings
    raw_df["question"] = raw_df["question"].fillna("").astype(str)
    raw_df["context"] = raw_df["context"].fillna("").astype(str)

    if debug:
        raw_df = raw_df.head(Config.DEBUG_SIZE)

    # 4. Process data
    if split == "train":
        processed_df = prepare_train_features(raw_df, tokenizer)
    else:
        processed_df = prepare_validation_features(raw_df, tokenizer)

    # 5. Save to cache
    print(f"Saving {split} features to {cache_path}...")
    # Use pyarrow engine for better handling of nested list columns
    processed_df.to_parquet(cache_path, engine="pyarrow")

    return processed_df
