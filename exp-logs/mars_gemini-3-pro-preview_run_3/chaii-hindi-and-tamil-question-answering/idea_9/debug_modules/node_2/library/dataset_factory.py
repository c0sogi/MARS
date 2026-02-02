import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.configuration import Config
from library.utils import load_data

# Disable tokenizer parallelism to avoid deadlocks in DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TAPTDataset(Dataset):
    """
    Dataset for Task-Adaptive Pretraining (Masked Language Modeling).
    Yields chunks of token IDs from the concatenated corpus.
    """

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings["input_ids"])


class QADataset(Dataset):
    """
    Dataset for Question Answering (Token Classification).
    Wraps processed features including input_ids, masks, labels, and metadata.
    """

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        # Convert core model inputs to tensors
        batch = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }

        # Add labels if they exist (Training/Validation)
        if item.get("labels") is not None:
            batch["labels"] = torch.tensor(item["labels"], dtype=torch.long)

        # Pass through metadata for inference/evaluation
        # These are not converted to tensors here; collate_fn handles them
        batch["offset_mapping"] = item["offset_mapping"]
        batch["example_id"] = item["example_id"]
        batch["sequence_ids"] = item["sequence_ids"]

        return batch


def qa_collate_fn(features):
    """
    Custom collate function to handle batching of tensors and metadata.
    Pads sequences and aggregates metadata into lists.
    """
    # Separate tensors and metadata
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]

    # Pad sequences
    # XLM-R pad token is 1
    batch = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=1
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        ),
    }

    # Pad labels if present
    if "labels" in features[0]:
        labels = [f["labels"] for f in features]
        # Pad labels with -100 (standard ignore index for CrossEntropyLoss)
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

    # Aggregate metadata as lists
    batch["offset_mapping"] = [f["offset_mapping"] for f in features]
    batch["example_id"] = [f["example_id"] for f in features]
    batch["sequence_ids"] = [f["sequence_ids"] for f in features]

    return batch


def prepare_tapt_data(tokenizer, load_cached_data=True):
    """
    Prepares data for Task-Adaptive Pretraining (MLM).
    Concatenates context from train, val, and test, then chunks into blocks.
    """
    cache_path = os.path.join(Config.TAPT_CACHE_DIR, "corpus.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached TAPT data from {cache_path}")
        df = pd.read_parquet(cache_path)
        encodings = {
            "input_ids": df["input_ids"].tolist(),
            "attention_mask": df["attention_mask"].tolist(),
        }
        return TAPTDataset(encodings)

    # 2. Process from scratch
    print("Preparing TAPT data from scratch...")

    # Load all splits to maximize domain coverage
    train_df = load_data("train")
    val_df = load_data("val")
    test_df = load_data("test")

    # Extract unique contexts
    texts = (
        pd.concat([train_df["context"], val_df["context"], test_df["context"]])
        .unique()
        .tolist()
    )
    texts = [t for t in texts if isinstance(t, str) and len(t) > 0]

    # Tokenize
    # We tokenize without truncation first to preserve all text
    tokenized_datasets = tokenizer(
        texts,
        return_special_tokens_mask=True,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )

    # Chunking logic
    block_size = Config.MAX_LENGTH
    concatenated_ids = []
    concatenated_mask = []

    for ids, mask in zip(
        tokenized_datasets["input_ids"], tokenized_datasets["attention_mask"]
    ):
        concatenated_ids.extend(ids)
        concatenated_mask.extend(mask)

    total_length = len(concatenated_ids)
    # Truncate to multiple of block_size
    total_length = (total_length // block_size) * block_size

    if total_length == 0:
        raise ValueError("Not enough text data for TAPT.")

    # Split into chunks
    result = {
        "input_ids": [
            concatenated_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ],
        "attention_mask": [
            concatenated_mask[i : i + block_size]
            for i in range(0, total_length, block_size)
        ],
    }

    # 3. Save to cache
    df_cache = pd.DataFrame(result)
    df_cache.to_parquet(cache_path)

    return TAPTDataset(result)


def process_qa_examples(examples, tokenizer, is_training=True):
    """
    Core logic for processing QA examples into sliding window features with Soft Overlap labels.
    """
    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        example_id = examples["id"][sample_index]

        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
            "example_id": example_id,
            "sequence_ids": sequence_ids,
        }

        if is_training:
            # Create labels: 0=O, 1=B-ANS, 2=I-ANS
            labels = [0] * len(input_ids)

            answer_text = examples["answer_text"][sample_index]
            answer_start = examples["answer_start"][sample_index]
            answer_end = answer_start + len(answer_text)

            # Iterate tokens to assign labels based on Soft Overlap
            for idx, (seq_id, (start_char, end_char)) in enumerate(
                zip(sequence_ids, offsets)
            ):
                # Skip question tokens (seq_id=0) and special tokens (seq_id=None)
                if seq_id != 1:
                    labels[idx] = -100
                    continue

                # Skip zero-length tokens (rare special tokens inside context)
                if start_char == end_char:
                    labels[idx] = -100
                    continue

                # Check intersection between token span and answer span
                overlap_start = max(start_char, answer_start)
                overlap_end = min(end_char, answer_end)

                if overlap_start < overlap_end:
                    # Overlap exists
                    if start_char <= answer_start < end_char:
                        # Token contains the start of the answer -> B-ANS
                        labels[idx] = 1
                    else:
                        # Token overlaps but isn't the start -> I-ANS
                        labels[idx] = 2
                else:
                    # No overlap -> O
                    labels[idx] = 0

            feature["labels"] = labels

        features.append(feature)

    return features


def prepare_qa_data(tokenizer, split="train", load_cached_data=True):
    """
    Prepares data for QA fine-tuning.
    Handles loading raw data, processing features, and caching.
    """
    cache_filename = f"{split}_features.parquet"
    cache_path = os.path.join(Config.QA_CACHE_DIR, cache_filename)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached QA features for {split} from {cache_path}")
        df = pd.read_parquet(cache_path)
        features = df.to_dict("records")

        # Restore None values in sequence_ids (stored as -1 in parquet)
        for f in features:
            if "sequence_ids" in f:
                f["sequence_ids"] = [None if x == -1 else x for x in f["sequence_ids"]]

        return QADataset(features)

    # 2. Process from scratch
    print(f"Preparing QA features for {split} from scratch...")
    df_raw = load_data(split)

    # Convert dataframe to dict of lists for batch processing
    examples = df_raw.to_dict("list")

    is_training = split in ["train", "val"]
    features = process_qa_examples(examples, tokenizer, is_training=is_training)

    # 3. Save to cache
    # Prepare for storage: handle None in sequence_ids
    features_for_df = []
    for f in features:
        f_copy = f.copy()
        f_copy["sequence_ids"] = [-1 if s is None else s for s in f["sequence_ids"]]
        features_for_df.append(f_copy)

    df_cache = pd.DataFrame(features_for_df)
    df_cache.to_parquet(cache_path)

    return QADataset(features)


def get_tokenizer():
    """
    Factory method to get the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
