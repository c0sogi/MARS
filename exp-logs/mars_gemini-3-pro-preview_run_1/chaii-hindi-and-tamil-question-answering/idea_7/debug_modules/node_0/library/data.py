import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from library.config import Config


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles both training (with labels) and testing (with mapping info) data.
    """

    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        if not self.is_test:
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)
            item["target_mapping"] = torch.tensor(
                row["target_mapping"], dtype=torch.float
            )
        else:
            item["example_id"] = row["example_id"]
            # offset_mapping is stored as a list of tuples (start, end)
            item["offset_mapping"] = torch.tensor(
                row["offset_mapping"], dtype=torch.long
            )

            # sequence_ids contains None for special tokens, convert to -1 for tensor
            seq_ids = [x if x is not None else -1 for x in row["sequence_ids"]]
            item["sequence_ids"] = torch.tensor(seq_ids, dtype=torch.long)

        return item


def prepare_train_features(config, tokenizer, load_cached_data=True):
    """
    Prepares training features with sliding windows and hard negative mining.
    Merges train and val sets for full-data training.
    Caches the processed dataframe to disk.
    """
    cache_path = os.path.join(config.WORKING_DIR, "train_features.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training features from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return QADataset(df, is_test=False)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Processing training features from scratch...")

    # 2. Load and Merge Data
    train_path = os.path.join(config.METADATA_DIR, "train.csv")
    val_path = os.path.join(config.METADATA_DIR, "val.csv")

    # Fallback if metadata not present (though prompt says it is)
    if not os.path.exists(train_path):
        train_path = os.path.join(config.INPUT_DIR, "train.csv")
        # If using raw input, we might not have val split, but logic assumes metadata structure

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()

    # Concatenate for full data training
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Clean whitespace
    df["context"] = df["context"].apply(lambda x: " ".join(str(x).split()))
    df["question"] = df["question"].apply(lambda x: " ".join(str(x).split()))

    # 3. Tokenization with Sliding Window
    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=config.MAX_LENGTH,
        stride=config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    # Lists to store indices for sampling strategy
    positive_indices = []
    hard_negative_indices = []
    negative_indices = []

    # 4. Feature Labeling
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = df.iloc[sample_index]

        ans_text = answers["answer_text"]
        start_char = answers["answer_start"]
        end_char = start_char + len(ans_text)

        # Determine context token span
        # sequence_ids: 0=question, 1=context, None=special
        token_start_index = 0
        while (
            token_start_index < len(sequence_ids)
            and sequence_ids[token_start_index] != 1
        ):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if token_start_index > token_end_index:
            # No context in this window (should rarely happen with proper stride)
            negative_indices.append(len(features))
            features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_positions": 0,
                    "end_positions": 0,
                    "target_mapping": 0.0,
                }
            )
            continue

        # Character span of the context in this window
        context_start_char = offsets[token_start_index][0]
        context_end_char = offsets[token_end_index][1]

        # Check containment
        is_positive = (context_start_char <= start_char) and (
            end_char <= context_end_char
        )

        # Check partial overlap (Hard Negative)
        overlap_start = max(context_start_char, start_char)
        overlap_end = min(context_end_char, end_char)
        has_overlap = overlap_start < overlap_end

        start_position = 0
        end_position = 0
        target_mapping = 0.0

        if is_positive:
            # Find token indices for answer
            idx = token_start_index
            while idx <= token_end_index and offsets[idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = token_end_index
            while idx >= token_start_index and offsets[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

            target_mapping = 1.0
            positive_indices.append(len(features))

        elif has_overlap:
            # Boundary/Hard Negative
            hard_negative_indices.append(len(features))
        else:
            # Random Negative
            negative_indices.append(len(features))

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "target_mapping": target_mapping,
            }
        )

    # 5. Sampling Strategy
    # Ensure deterministic sampling
    random.seed(config.BASE_SEED)

    n_positives = len(positive_indices)
    n_negatives_needed = int(n_positives * config.NEGATIVE_SAMPLING_RATIO)

    selected_indices = positive_indices.copy()

    # Prioritize Hard Negatives
    if len(hard_negative_indices) >= n_negatives_needed:
        selected_indices.extend(
            random.sample(hard_negative_indices, n_negatives_needed)
        )
    else:
        selected_indices.extend(hard_negative_indices)
        remaining_needed = n_negatives_needed - len(hard_negative_indices)

        if len(negative_indices) > remaining_needed:
            selected_indices.extend(random.sample(negative_indices, remaining_needed))
        else:
            selected_indices.extend(negative_indices)

    final_features = [features[i] for i in selected_indices]

    # 6. Save and Return
    feature_df = pd.DataFrame(final_features)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    feature_df.to_parquet(cache_path)

    print(f"Generated {len(feature_df)} features.")
    print(
        f"Positives: {n_positives}, Hard Negatives: {len(hard_negative_indices)}, Total Negatives Used: {len(selected_indices) - n_positives}"
    )

    return QADataset(feature_df, is_test=False)


def prepare_test_features(config, tokenizer):
    """
    Prepares test features using exhaustive sliding windows.
    Preserves offset mapping and example IDs for post-processing.
    """
    test_path = os.path.join(config.METADATA_DIR, "test.csv")
    df = pd.read_csv(test_path)

    df["context"] = df["context"].apply(lambda x: " ".join(str(x).split()))
    df["question"] = df["question"].apply(lambda x: " ".join(str(x).split()))

    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=config.MAX_LENGTH,
        stride=config.DOC_STRIDE,
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
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        example_id = df.iloc[sample_index]["id"]

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,
                "example_id": example_id,
                "sequence_ids": sequence_ids,
            }
        )

    feature_df = pd.DataFrame(features)
    return QADataset(feature_df, is_test=True)
