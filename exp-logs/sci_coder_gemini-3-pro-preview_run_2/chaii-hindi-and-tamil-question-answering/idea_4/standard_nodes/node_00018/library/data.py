import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from library.config import Config


def prepare_train_features(examples, tokenizer):
    """
    Tokenizes training data with sliding windows and maps answer spans to token indices.
    """
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",  # Truncate context, not question
        max_length=Config.MAX_LEN,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=False,  # DistilBERT does not use token_type_ids (Cite solution_lesson_node_00016)
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = 0

        # Grab the sequence corresponding to that example
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several features.
        sample_index = sample_mapping[i]
        answers = examples["answer_text"][sample_index]
        start_char = examples["answer_start"][sample_index]

        # Calculate end character position
        end_char = start_char + len(answers)

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Move the token_start_index and token_end_index to the two ends of the answer.
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer):
    """
    Tokenizes validation/test data with sliding windows.
    Keeps offset_mapping and example_id for post-processing.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=Config.MAX_LEN,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=False,  # DistilBERT does not use token_type_ids
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles both training (returns labels) and inference (returns ids/offsets).
    """

    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode

        # Pre-convert columns to lists for faster access if it's a DataFrame
        if isinstance(data, pd.DataFrame):
            self.input_ids = data["input_ids"].tolist()
            self.attention_mask = data["attention_mask"].tolist()
            # self.token_type_ids removed for DistilBERT

            if self.mode == "train":
                self.start_positions = data["start_positions"].tolist()
                self.end_positions = data["end_positions"].tolist()
        else:
            # Assume dictionary or HF dataset structure
            self.input_ids = data["input_ids"]
            self.attention_mask = data["attention_mask"]
            # self.token_type_ids removed for DistilBERT
            if self.mode == "train":
                self.start_positions = data["start_positions"]
                self.end_positions = data["end_positions"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.mode == "train":
            item["start_positions"] = torch.tensor(
                self.start_positions[idx], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(
                self.end_positions[idx], dtype=torch.long
            )

        return item


def load_processed_data(tokenizer, split="train", load_cached_data=True):
    """
    Loads raw data, processes it (tokenization + sliding window), and caches the result.

    Args:
        tokenizer: The Hugging Face tokenizer.
        split: "train", "val", or "test".
        load_cached_data: Boolean to determine if cache should be used.

    Returns:
        pd.DataFrame: The processed features.
    """
    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split}_features.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from cache: {cache_path}")
        df = pd.read_parquet(cache_path)

        # Deserialize offset_mapping if it exists and is a string
        if "offset_mapping" in df.columns and df["offset_mapping"].dtype == object:
            # Check if it looks like a stringified JSON (basic heuristic)
            if isinstance(df["offset_mapping"].iloc[0], str):
                df["offset_mapping"] = df["offset_mapping"].apply(json.loads)

        return df

    # 2. Process from scratch
    print(f"Processing {split} data from scratch...")

    # Identify input file
    if split == "train":
        input_path = Config.TRAIN_PATH
    elif split == "val":
        input_path = Config.VAL_PATH
    elif split == "test":
        input_path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load raw CSV
    df_raw = pd.read_csv(input_path)

    # Debugging: Sample subset
    if Config.DEBUG:
        df_raw = df_raw.head(Config.DEBUG_SAMPLE_SIZE)

    # Convert to Hugging Face Dataset for efficient mapping
    hf_dataset = HFDataset.from_pandas(df_raw)

    # Apply tokenization
    if split == "train":
        processed_dataset = hf_dataset.map(
            lambda x: prepare_train_features(x, tokenizer),
            batched=True,
            remove_columns=hf_dataset.column_names,
            desc=f"Running tokenizer on {split} dataset",
        )
    else:
        processed_dataset = hf_dataset.map(
            lambda x: prepare_validation_features(x, tokenizer),
            batched=True,
            remove_columns=hf_dataset.column_names,
            desc=f"Running tokenizer on {split} dataset",
        )

    # Convert back to Pandas for caching
    df_processed = processed_dataset.to_pandas()

    # Handle serialization for Parquet compatibility
    # offset_mapping contains None values and tuples, so we serialize to JSON string
    if "offset_mapping" in df_processed.columns:

        def deep_tolist(x):
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if isinstance(x, (list, tuple)):
                return [deep_tolist(i) for i in x]
            return x

        df_processed["offset_mapping"] = df_processed["offset_mapping"].apply(
            lambda x: json.dumps(deep_tolist(x))
        )

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_processed.to_parquet(cache_path, index=False)
    print(f"Saved processed {split} features to {cache_path}")

    # Deserialize before returning
    if "offset_mapping" in df_processed.columns:
        df_processed["offset_mapping"] = df_processed["offset_mapping"].apply(
            json.loads
        )

    return df_processed
