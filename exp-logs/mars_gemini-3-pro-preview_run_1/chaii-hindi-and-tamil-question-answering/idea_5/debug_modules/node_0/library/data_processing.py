import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.configuration import Config
from library.utilities import set_seed


class QADataset(Dataset):
    """
    PyTorch Dataset for the Question Answering task.
    Handles inputs for both the Span Prediction head and the Relevance Classification head.
    """

    def __init__(self, features, mode="train"):
        self.mode = mode
        self.input_ids = features["input_ids"]
        self.attention_mask = features["attention_mask"]

        # Offset mapping and example_id are needed for inference reconstruction
        self.offset_mapping = features.get("offset_mapping", None)
        self.example_id = features.get("example_id", None)
        self.sequence_ids = features.get("sequence_ids", None)

        if self.mode == "train":
            self.start_positions = features["start_positions"]
            self.end_positions = features["end_positions"]
            self.relevance_labels = features["relevance_labels"]

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
            item["relevance_labels"] = torch.tensor(
                self.relevance_labels[idx], dtype=torch.float
            )
        else:
            # For inference, we might need these to map back to text
            if self.offset_mapping is not None:
                item["offset_mapping"] = torch.tensor(
                    self.offset_mapping[idx], dtype=torch.long
                )
            if self.example_id is not None:
                item["example_id"] = self.example_id[idx]  # Keep as string
            if self.sequence_ids is not None:
                # Sequence IDs often contain None, handle carefully if converting to tensor
                # Usually we handle sequence_ids logic outside the tensor conversion for inference
                pass

        return item


def prepare_train_features(examples, tokenizer, config):
    """
    Tokenizes examples, applies sliding window, generates labels, and performs negative sampling.
    """
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    # This results in multiple features when a context is long.
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=config.max_length,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Storage for processed features
    features = {
        "input_ids": [],
        "attention_mask": [],
        "start_positions": [],
        "end_positions": [],
        "relevance_labels": [],
    }

    # Temporary lists to separate positives and negatives for sampling
    positive_indices = []
    negative_indices = []

    # We need to access the original data to get answers
    # Convert dataframe columns to lists for faster access
    answer_starts = examples["answer_start"].tolist()
    answer_texts = examples["answer_text"].tolist()

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give multiple spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer_start_char = answer_starts[sample_index]
        answer_text = answer_texts[sample_index]
        answer_end_char = answer_start_char + len(answer_text)

        # Find the start and end of the context in the current window
        # sequence_ids: 0 for question, 1 for context, None for special tokens
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Detect if the answer is fully inside the context of this window
        # offsets[x] returns (start_char, end_char)

        # If the answer is not fully inside the context, label is (0, 0)
        # We check if the context window covers the answer span
        if not (
            offsets[context_start][0] <= answer_start_char
            and offsets[context_end][1] >= answer_end_char
        ):
            start_position = 0
            end_position = 0
            relevance_label = 0.0
            is_positive = False
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= answer_start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= answer_end_char:
                idx -= 1
            end_position = idx + 1

            relevance_label = 1.0
            is_positive = True

        # Store feature data temporarily
        feature_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_position,
            "end_positions": end_position,
            "relevance_labels": relevance_label,
        }

        # Segregate for sampling
        if is_positive:
            features["input_ids"].append(feature_data["input_ids"])
            features["attention_mask"].append(feature_data["attention_mask"])
            features["start_positions"].append(feature_data["start_positions"])
            features["end_positions"].append(feature_data["end_positions"])
            features["relevance_labels"].append(feature_data["relevance_labels"])
            positive_indices.append(len(features["input_ids"]) - 1)
        else:
            # For negatives, we just store the data in a separate list first
            negative_indices.append(feature_data)

    # Negative Sampling Logic
    num_positives = len(positive_indices)
    num_negatives_to_keep = int(num_positives * config.negative_sampling_ratio)

    # If we have negatives, sample them
    if len(negative_indices) > 0:
        # Ensure we don't try to sample more than we have
        num_negatives_to_keep = min(num_negatives_to_keep, len(negative_indices))

        # Randomly select negatives
        rng = np.random.default_rng(config.seed)
        selected_negative_indices = rng.choice(
            len(negative_indices), size=num_negatives_to_keep, replace=False
        )

        for idx in selected_negative_indices:
            neg_feat = negative_indices[idx]
            features["input_ids"].append(neg_feat["input_ids"])
            features["attention_mask"].append(neg_feat["attention_mask"])
            features["start_positions"].append(neg_feat["start_positions"])
            features["end_positions"].append(neg_feat["end_positions"])
            features["relevance_labels"].append(neg_feat["relevance_labels"])

    # Shuffle the combined dataset
    # We do this by creating a permutation of indices
    total_len = len(features["input_ids"])
    perm = np.random.permutation(total_len)

    final_features = {
        "input_ids": [features["input_ids"][i] for i in perm],
        "attention_mask": [features["attention_mask"][i] for i in perm],
        "start_positions": [features["start_positions"][i] for i in perm],
        "end_positions": [features["end_positions"][i] for i in perm],
        "relevance_labels": [features["relevance_labels"][i] for i in perm],
    }

    return final_features


def prepare_test_features(examples, tokenizer, config):
    """
    Tokenizes test examples with sliding window.
    Keeps offset mapping and example IDs for post-processing.
    """
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=config.max_length,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    features = {
        "input_ids": [],
        "attention_mask": [],
        "offset_mapping": [],
        "example_id": [],
        "sequence_ids": [],
    }

    example_ids = examples["id"].tolist()

    for i, input_ids in enumerate(tokenized_examples["input_ids"]):
        sample_index = sample_mapping[i]

        features["input_ids"].append(input_ids)
        features["attention_mask"].append(tokenized_examples["attention_mask"][i])
        features["offset_mapping"].append(tokenized_examples["offset_mapping"][i])
        features["example_id"].append(example_ids[sample_index])

        # We need sequence_ids to distinguish question from context during inference
        # sequence_ids() returns a list with None, 0, 1. We replace None with -1 for simpler storage if needed,
        # but here we keep it as list to be handled in collate or post-processing.
        # However, Parquet/Arrays don't like None mixed with Ints.
        # We will replace None with -1.
        seq_ids = tokenized_examples.sequence_ids(i)
        seq_ids_clean = [-1 if s is None else s for s in seq_ids]
        features["sequence_ids"].append(seq_ids_clean)

    return features


def get_train_data(config, load_cached_data=True):
    """
    Loads training data, merges train/val splits, processes features, and returns a Dataset.
    Implements caching to Parquet.
    """
    set_seed(config.seed)

    # Check cache
    if load_cached_data and os.path.exists(config.train_features_path):
        try:
            df = pd.read_parquet(config.train_features_path)
            # Convert back to dict of lists for Dataset
            features = {
                "input_ids": df["input_ids"].tolist(),
                "attention_mask": df["attention_mask"].tolist(),
                "start_positions": df["start_positions"].tolist(),
                "end_positions": df["end_positions"].tolist(),
                "relevance_labels": df["relevance_labels"].tolist(),
            }
            return QADataset(features, mode="train")
        except Exception as e:
            print(f"Failed to load cached train data: {e}. Recomputing...")

    # Load and Merge Data
    train_df = pd.read_csv(config.train_meta_path)
    val_df = pd.read_csv(config.val_meta_path)
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Ensure text columns are strings
    combined_df["question"] = combined_df["question"].astype(str)
    combined_df["context"] = combined_df["context"].astype(str)
    combined_df["answer_text"] = combined_df["answer_text"].astype(str)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Process Features
    features = prepare_train_features(combined_df, tokenizer, config)

    # Save to Cache
    # Convert to DataFrame for Parquet
    # Note: PyArrow handles columns of lists efficiently
    df_out = pd.DataFrame(features)
    # Ensure types are compatible
    # input_ids, attention_mask are lists of ints.
    # start/end are ints. relevance is float.
    os.makedirs(os.path.dirname(config.train_features_path), exist_ok=True)
    df_out.to_parquet(config.train_features_path, index=False)

    return QADataset(features, mode="train")


def get_test_data(config, load_cached_data=True):
    """
    Loads test data, processes features, and returns a Dataset.
    Implements caching to Parquet.
    """
    set_seed(config.seed)

    if load_cached_data and os.path.exists(config.test_features_path):
        try:
            df = pd.read_parquet(config.test_features_path)
            features = {
                "input_ids": df["input_ids"].tolist(),
                "attention_mask": df["attention_mask"].tolist(),
                "offset_mapping": df["offset_mapping"].tolist(),
                "example_id": df["example_id"].tolist(),
                "sequence_ids": df["sequence_ids"].tolist(),
            }
            return QADataset(features, mode="test")
        except Exception as e:
            print(f"Failed to load cached test data: {e}. Recomputing...")

    # Load Data
    test_df = pd.read_csv(config.test_meta_path)
    test_df["question"] = test_df["question"].astype(str)
    test_df["context"] = test_df["context"].astype(str)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Process Features
    features = prepare_test_features(test_df, tokenizer, config)

    # Save to Cache
    df_out = pd.DataFrame(features)
    os.makedirs(os.path.dirname(config.test_features_path), exist_ok=True)
    df_out.to_parquet(config.test_features_path, index=False)

    return QADataset(features, mode="test")
