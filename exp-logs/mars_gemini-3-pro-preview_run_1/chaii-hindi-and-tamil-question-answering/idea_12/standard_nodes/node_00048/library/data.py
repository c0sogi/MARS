import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    """

    def __init__(self, features, mode="train"):
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        # Convert lists to tensors
        input_ids = torch.tensor(feature["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(feature["attention_mask"], dtype=torch.long)

        if self.mode == "train":
            start_position = torch.tensor(feature["start_position"], dtype=torch.long)
            end_position = torch.tensor(feature["end_position"], dtype=torch.long)
            relevance_label = torch.tensor(
                feature["relevance_label"], dtype=torch.float
            )

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "relevance_labels": relevance_label,
            }
        else:
            # For inference, we might need the example_id and offsets to reconstruct the answer
            # We return them as part of the item, but the collate_fn or loop will handle them.
            # Since standard DataLoader collates tensors, we keep non-tensors out or handle carefully.
            # Here we return indices and let the inference loop access the raw feature list for metadata if needed,
            # or return the metadata if the batch size is 1 or custom collate is used.
            # To be safe for standard batching, we return the index.
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "feature_idx": torch.tensor(idx, dtype=torch.long),
            }


def prepare_train_features(cfg, examples, tokenizer):
    """
    Tokenizes training data with sliding windows, generates labels,
    and performs negative sampling.
    """
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    # This results in one example possible giving several features when a context is long.
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",  # Truncate context, not question
        max_length=cfg.max_len,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    # Convert dataframe columns to lists for faster access
    answer_starts = examples["answer_start"].tolist()
    answer_texts = examples["answer_text"].tolist()

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]

        # Sequence ids help us distinguish question from context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Start/End character index of the answer in the text
        start_char = answer_starts[sample_index]
        end_char = start_char + len(answer_texts[sample_index])

        # Find the start and end of the context in the input_ids
        # sequence_ids is None for special tokens, 0 for question, 1 for context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in this specific window)
        # The offsets are (start_char, end_char) for each token
        # If the window does not contain the answer, label as CLS (0)

        # Check if the answer is fully inside the context of this feature
        # offsets[token_start_index][0] is the start char of the first context token
        # offsets[token_end_index][1] is the end char of the last context token

        if not (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            start_position = 0
            end_position = 0
            relevance_label = 0.0
        else:
            # Move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could use binary search, but linear is fine for small window size.

            # Find start token
            current_idx = token_start_index
            while (
                current_idx <= token_end_index and offsets[current_idx][0] <= start_char
            ):
                current_idx += 1
            start_position = current_idx - 1

            # Find end token
            current_idx = token_end_index
            while (
                current_idx >= token_start_index and offsets[current_idx][1] >= end_char
            ):
                current_idx -= 1
            end_position = current_idx + 1

            relevance_label = 1.0

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_position": start_position,
                "end_position": end_position,
                "relevance_label": relevance_label,
            }
        )

    # Negative Sampling
    # Separate positives and negatives
    pos_features = [f for f in features if f["relevance_label"] > 0.5]
    neg_features = [f for f in features if f["relevance_label"] < 0.5]

    # Calculate number of negatives to keep
    n_pos = len(pos_features)
    n_neg_keep = int(n_pos * cfg.neg_pos_ratio)

    # Sample negatives
    # Ensure reproducibility
    set_seed(cfg.seeds[0])
    if len(neg_features) > n_neg_keep:
        import random

        sampled_neg_features = random.sample(neg_features, n_neg_keep)
    else:
        sampled_neg_features = neg_features

    # Combine and shuffle
    final_features = pos_features + sampled_neg_features
    import random

    random.shuffle(final_features)

    return final_features


def prepare_test_features(cfg, examples, tokenizer):
    """
    Tokenizes test data with exhaustive sliding windows.
    Stores metadata for reconstruction.
    """
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=cfg.max_len,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []
    example_ids = examples["id"].tolist()
    contexts = examples["context"].tolist()

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sample_index = sample_mapping[i]

        # Sanitize offset_mapping: convert tuples to lists for Parquet compatibility
        # offsets is a list of (start, end) tuples.
        sanitized_offsets = [list(o) for o in offsets]

        # We need sequence_ids to know which tokens are context during inference
        sequence_ids = tokenized_examples.sequence_ids(i)
        # Replace None with -1 for serialization
        sanitized_sequence_ids = [s if s is not None else -1 for s in sequence_ids]

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "example_id": example_ids[sample_index],
                "offset_mapping": sanitized_offsets,
                "sequence_ids": sanitized_sequence_ids,
                "context": contexts[
                    sample_index
                ],  # Store context for easy extraction later
            }
        )

    return features


def get_data(cfg: Config, load_cached_data: bool = True):
    """
    Main function to load and process data.
    Handles caching to disk to save time on restarts.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # -------------------------------------------------------------------------
    # 1. Prepare Training Data (Merged Train + Val)
    # -------------------------------------------------------------------------
    train_cache_path = os.path.join(cfg.cache_dir, "train_features.parquet")

    if load_cached_data and os.path.exists(train_cache_path):
        print(f"Loading cached training features from {train_cache_path}...")
        train_df = pd.read_parquet(train_cache_path)
        # Convert back to list of dicts
        train_features = train_df.to_dict("records")
    else:
        print("Processing training data from scratch...")
        # Load metadata
        train_meta = pd.read_csv(os.path.join(cfg.metadata_dir, "train.csv"))
        val_meta = pd.read_csv(os.path.join(cfg.metadata_dir, "val.csv"))

        # Merge for full training
        full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

        # Process
        train_features = prepare_train_features(cfg, full_train_meta, tokenizer)

        # Cache
        print(f"Saving training features to {train_cache_path}...")
        train_df = pd.DataFrame(train_features)
        train_df.to_parquet(train_cache_path)

    # -------------------------------------------------------------------------
    # 2. Prepare Test Data
    # -------------------------------------------------------------------------
    test_cache_path = os.path.join(cfg.cache_dir, "test_features.parquet")

    if load_cached_data and os.path.exists(test_cache_path):
        print(f"Loading cached test features from {test_cache_path}...")
        test_df = pd.read_parquet(test_cache_path)
        test_features = test_df.to_dict("records")
    else:
        print("Processing test data from scratch...")
        test_meta = pd.read_csv(os.path.join(cfg.metadata_dir, "test.csv"))

        # Process
        test_features = prepare_test_features(cfg, test_meta, tokenizer)

        # Cache
        print(f"Saving test features to {test_cache_path}...")
        test_df = pd.DataFrame(test_features)
        test_df.to_parquet(test_cache_path)

    # -------------------------------------------------------------------------
    # 3. Create Datasets
    # -------------------------------------------------------------------------
    train_dataset = QADataset(train_features, mode="train")
    test_dataset = QADataset(test_features, mode="test")

    # For test features, we also return the raw list because we need metadata (offsets, ids)
    # that the Dataset/DataLoader might obscure or complicate.
    return train_dataset, test_dataset, test_features
