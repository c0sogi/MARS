import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles both training data (with targets) and test data (without targets).
    """

    def __init__(self, features, mode="train"):
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        # Common inputs
        item = {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
        }

        # Training targets
        if self.mode == "train":
            item["start_positions"] = torch.tensor(
                feature["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(
                feature["end_positions"], dtype=torch.long
            )
            item["answerable_label"] = torch.tensor(
                feature["answerable_label"], dtype=torch.float
            )

        # Inference metadata (only needed for post-processing, but kept in dataset for alignment)
        # Note: Strings/metadata are usually handled outside the DataLoader collation or via custom collate,
        # but for simple iteration we can return them or access them via the dataframe directly.
        # Here we focus on tensor data for the model.
        if self.mode == "test":
            # For test, we might need example_id and offset_mapping for post-processing.
            # However, standard PyTorch collate_fn fails with lists/strings mixed with tensors.
            # We will rely on the features list being aligned with the dataloader order.
            pass

        return item


def prepare_train_features(df, tokenizer=None, load_cached_data=True):
    """
    Tokenizes training data with sliding window and maps answers to token positions.
    Implements caching to Parquet.
    """
    # Determine cache path based on debug mode
    suffix = "_debug" if Config.debug else ""
    cache_path = os.path.join(
        Config.working_dir, f"cached_train_features{suffix}.parquet"
    )

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training features from {cache_path}...")
        features_df = pd.read_parquet(cache_path)
        # Convert dataframe back to list of dicts
        features = features_df.to_dict("records")
        return features

    print("Processing training features from scratch...")

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Handle Debug Mode
    if Config.debug:
        df = df.head(Config.debug_sample_size).copy()

    # Clean data
    df["question"] = df["question"].astype(str).fillna("")
    df["context"] = df["context"].astype(str).fillna("")
    df["answer_text"] = df["answer_text"].astype(str).fillna("")

    # Tokenization with sliding window
    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=Config.max_len,
        stride=Config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # CLS token index (usually 0 for XLM-R)
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence ids to distinguish question from context
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        row = df.iloc[sample_index]

        answer_text = row["answer_text"]
        start_char = row["answer_start"]
        end_char = start_char + len(answer_text)

        # Find the start and end of the context in the current window
        # sequence_ids: 0 for question, 1 for context, None for special tokens
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span
        # If the context window does not fully contain the answer, label as 0 (unanswerable in this window)
        if not (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            start_position = cls_index
            end_position = cls_index
            answerable_label = 0.0
        else:
            # Move the token_start_index and token_end_index to the two ends of the answer
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            start_position = token_start_index - 1

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_position = token_end_index + 1

            answerable_label = 1.0

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "answerable_label": answerable_label,
                "example_id": row["id"],  # Useful for debugging/grouping
            }
        )

    # Cache the results
    # We convert to DataFrame for easy Parquet storage
    # Parquet handles lists in columns well
    features_df = pd.DataFrame(features)

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)
    features_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(features)} training features to {cache_path}")

    return features


def prepare_test_features(df, tokenizer=None, load_cached_data=True):
    """
    Tokenizes test data with sliding window.
    Preserves offset mappings for post-processing.
    Implements caching to Parquet.
    """
    # Determine cache path based on debug mode
    suffix = "_debug" if Config.debug else ""
    cache_path = os.path.join(
        Config.working_dir, f"cached_test_features{suffix}.parquet"
    )

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test features from {cache_path}...")
        features_df = pd.read_parquet(cache_path)
        features = features_df.to_dict("records")
        return (
            features,
            features_df,
        )  # Return DF as well for easy metadata access during inference

    print("Processing test features from scratch...")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    if Config.debug:
        df = df.head(Config.debug_sample_size).copy()

    df["question"] = df["question"].astype(str).fillna("")
    df["context"] = df["context"].astype(str).fillna("")

    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=Config.max_len,
        stride=Config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        sample_index = sample_mapping[i]
        example_id = df.iloc[sample_index]["id"]

        # Identify context tokens for valid prediction masking later
        sequence_ids = tokenized_examples.sequence_ids(i)

        # We store offset_mapping as a list of lists/tuples for retrieval
        # However, Parquet might have issues with list of tuples, so we convert to list of lists
        offsets_list = [list(o) for o in offsets]

        # Replace None in sequence_ids with -1 for serialization safety
        seq_ids_clean = [s if s is not None else -1 for s in sequence_ids]

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "example_id": example_id,
                "offset_mapping": offsets_list,
                "sequence_ids": seq_ids_clean,
                "context": df.iloc[sample_index][
                    "context"
                ],  # Store context for extraction
            }
        )

    features_df = pd.DataFrame(features)

    os.makedirs(Config.working_dir, exist_ok=True)
    features_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(features)} test features to {cache_path}")

    return features, features_df
