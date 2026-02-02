import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed

# Constants for BIO tagging
LABEL_TO_ID = {"O": 0, "B-ANS": 1, "I-ANS": 2}
ID_TO_LABEL = {0: "O", 1: "B-ANS", 2: "I-ANS"}


def get_tokenizer():
    """
    Loads the tokenizer defined in the configuration.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)


class TAPTDataset(Dataset):
    """
    Dataset for Task-Adaptive Pretraining (Masked Language Modeling).
    Processes raw text from contexts using sliding windows to maximize coverage.
    """

    def __init__(self, texts, tokenizer):
        self.examples = []

        # Tokenize with sliding window to ensure long contexts are fully covered
        # Using the same stride/length as QA to align domains
        tokenized = tokenizer(
            texts,
            max_length=Config.MAX_LENGTH,
            truncation=True,
            stride=Config.DOC_STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
            return_special_tokens_mask=True,
        )

        # Store input_ids and attention_mask
        # Masking is handled dynamically by the DataCollator
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        for i in range(len(input_ids)):
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids[i], dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask[i], dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QADataset(Dataset):
    """
    Dataset for Question Answering (Token Classification).
    Wraps processed features (input_ids, labels, metadata).
    """

    def __init__(self, features_df):
        self.features = features_df.to_dict("records")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        # Convert list features back to tensors
        sample = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }

        # Add labels if they exist (Train/Val)
        if "labels" in item and item["labels"] is not None:
            # Handle potential NaN or None in parquet loading
            labels = item["labels"]
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            sample["labels"] = torch.tensor(labels, dtype=torch.long)

        # Pass through metadata
        # Note: offset_mapping, example_id, context, sequence_ids are kept as standard types
        # and handled by the custom collate_fn
        if "offset_mapping" in item:
            sample["offset_mapping"] = item["offset_mapping"]
        if "example_id" in item:
            sample["example_id"] = item["example_id"]
        if "context" in item:
            sample["context"] = item["context"]
        if "sequence_ids" in item:
            sample["sequence_ids"] = item["sequence_ids"]

        return sample


def qa_collate_fn(batch):
    """
    Custom collate function to handle mixed tensor and metadata fields.
    """
    # Stack standard tensors
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    batch_out = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Stack labels if present
    if "labels" in batch[0]:
        batch_out["labels"] = torch.stack([b["labels"] for b in batch])

    # Collect metadata as lists
    if "offset_mapping" in batch[0]:
        batch_out["offset_mapping"] = [b["offset_mapping"] for b in batch]
    if "example_id" in batch[0]:
        batch_out["example_id"] = [b["example_id"] for b in batch]
    if "context" in batch[0]:
        batch_out["context"] = [b["context"] for b in batch]
    if "sequence_ids" in batch[0]:
        batch_out["sequence_ids"] = [b["sequence_ids"] for b in batch]

    return batch_out


def prepare_tapt_data(tokenizer):
    """
    Prepares data for Task-Adaptive Pretraining.
    Aggregates contexts from Train and Test sets.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Extract unique contexts to avoid duplication bias
    # (Though TAPT usually benefits from seeing distribution, unique contexts is safer for small data)
    train_contexts = train_df["context"].dropna().unique().tolist()
    test_contexts = test_df["context"].dropna().unique().tolist()

    all_contexts = train_contexts + test_contexts
    print(f"TAPT: Loaded {len(all_contexts)} unique contexts.")

    return TAPTDataset(all_contexts, tokenizer)


def _process_qa_batch(batch_df, tokenizer, is_training=True):
    """
    Internal function to process a batch of raw QA examples into features.
    Implements Sliding Window and Strict Containment logic.
    """
    questions = [str(q).strip() for q in batch_df["question"]]
    contexts = [str(c) for c in batch_df["context"]]
    ids = batch_df["id"].tolist()

    # Tokenize with sliding window
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        example_id = ids[sample_idx]
        context_text = contexts[sample_idx]

        # Get sequence IDs (0 for question, 1 for context, None for special)
        # We replace None with -1 for serialization safety
        seq_ids = tokenized.sequence_ids(i)
        seq_ids_clean = [s if s is not None else -1 for s in seq_ids]

        feature = {
            "input_ids": tokenized["input_ids"][i],
            "attention_mask": tokenized["attention_mask"][i],
            "offset_mapping": offsets,  # List of [start, end]
            "example_id": example_id,
            "context": context_text,
            "sequence_ids": seq_ids_clean,
        }

        if is_training:
            # Label generation logic
            answer_text = str(batch_df.iloc[sample_idx]["answer_text"])
            answer_start = int(batch_df.iloc[sample_idx]["answer_start"])
            answer_end = answer_start + len(answer_text)

            # Initialize labels as O (0)
            labels = [LABEL_TO_ID["O"]] * len(feature["input_ids"])

            # Determine context bounds in tokens
            # sequence_ids: 0=question, 1=context, None=special
            try:
                token_start_index = seq_ids.index(1)
                # Find last context token
                token_end_index = len(seq_ids) - 1
                while seq_ids[token_end_index] != 1:
                    token_end_index -= 1
            except ValueError:
                # No context tokens (rare edge case with very long questions)
                token_start_index = 0
                token_end_index = 0

            # STRICT CONTAINMENT CHECK
            # Check if the window fully contains the answer span
            # offsets[i] is (start_char, end_char)

            window_start_char = offsets[token_start_index][0]
            window_end_char = offsets[token_end_index][1]

            if not (
                window_start_char <= answer_start and window_end_char >= answer_end
            ):
                # Answer not fully contained -> All O
                pass
            else:
                # Answer is contained. Map char positions to tokens.
                # We use the char_to_token method of the encoding object if available,
                # but here we have the batch encoding.
                # Since we are iterating, we can manually map using offsets for precision.

                start_token_idx = -1
                end_token_idx = -1

                # Find start token: first token where end_char > answer_start
                # (and is within context)
                for idx in range(token_start_index, token_end_index + 1):
                    # Token span: [s, e)
                    # We want the token that contains the start character
                    if offsets[idx][0] <= answer_start < offsets[idx][1]:
                        start_token_idx = idx
                        break

                # Find end token: token that contains the last character of answer (answer_end - 1)
                for idx in range(token_start_index, token_end_index + 1):
                    if offsets[idx][0] <= (answer_end - 1) < offsets[idx][1]:
                        end_token_idx = idx
                        break

                if start_token_idx != -1 and end_token_idx != -1:
                    labels[start_token_idx] = LABEL_TO_ID["B-ANS"]
                    for k in range(start_token_idx + 1, end_token_idx + 1):
                        labels[k] = LABEL_TO_ID["I-ANS"]

            feature["labels"] = labels

        features.append(feature)

    return features


def get_qa_data(tokenizer, load_cached_data=True):
    """
    Main entry point for QA data.
    Loads Train/Val/Test data, processes into features (or loads from cache),
    and returns QADataset objects.
    """
    set_seed(Config.SEED_LIST[0])

    # Ensure cache directory exists
    os.makedirs(Config.QA_CACHE_DIR, exist_ok=True)

    # Define paths
    train_cache = os.path.join(Config.QA_CACHE_DIR, "train_features.parquet")
    val_cache = os.path.join(Config.QA_CACHE_DIR, "val_features.parquet")
    test_cache = os.path.join(Config.QA_CACHE_DIR, "test_features.parquet")

    datasets = {}

    # Helper to process or load
    def _get_dataset(split_name, meta_path, cache_path, is_training):
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split_name} features from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            # Parquet might load lists as numpy arrays, ensure compatibility
            return QADataset(df)

        print(f"Processing {split_name} data from scratch...")
        meta_df = pd.read_csv(meta_path)

        # Process in chunks to avoid massive memory usage during tokenization
        # though dataset is small enough (1k rows) to do in one go.
        features = _process_qa_batch(meta_df, tokenizer, is_training=is_training)

        # Convert to DataFrame for caching
        feature_df = pd.DataFrame(features)

        # Save to parquet
        # PyArrow handles nested lists (input_ids, offset_mapping) well
        feature_df.to_parquet(cache_path, index=False)

        return QADataset(feature_df)

    # 1. Train
    datasets["train"] = _get_dataset(
        "train", Config.TRAIN_META_PATH, train_cache, is_training=True
    )

    # 2. Val
    datasets["val"] = _get_dataset(
        "val", Config.VAL_META_PATH, val_cache, is_training=True
    )

    # 3. Test
    datasets["test"] = _get_dataset(
        "test", Config.TEST_META_PATH, test_cache, is_training=False
    )

    return datasets["train"], datasets["val"], datasets["test"]
