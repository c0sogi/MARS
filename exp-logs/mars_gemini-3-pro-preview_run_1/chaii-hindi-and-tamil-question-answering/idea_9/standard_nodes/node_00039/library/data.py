import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from library.config import Config
from library.utils import set_seed


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        start_positions=None,
        end_positions=None,
        relevance_labels=None,
    ):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        self.start_positions = None
        self.end_positions = None
        self.relevance_labels = None

        if start_positions is not None:
            self.start_positions = torch.tensor(start_positions, dtype=torch.long)
            self.end_positions = torch.tensor(end_positions, dtype=torch.long)
            self.relevance_labels = torch.tensor(relevance_labels, dtype=torch.float)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

        if self.start_positions is not None:
            item["start_positions"] = self.start_positions[idx]
            item["end_positions"] = self.end_positions[idx]
            item["relevance_labels"] = self.relevance_labels[idx]

        return item


def prepare_train_features(config, tokenizer, load_cached_data=True):
    """
    Prepares training features with sliding windows, hard negative mining, and caching.

    Strategy:
    1. Load and merge train/val metadata.
    2. Tokenize with sliding windows.
    3. Identify Positives (contain full answer).
    4. Identify Boundary Negatives (adjacent to positives) and Random Negatives.
    5. Sample Negatives to maintain 2:1 ratio.
    6. Cache results to Parquet.
    """
    cache_path = os.path.join(config.working_dir, "train_features.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training features from {cache_path}")
        df = pd.read_parquet(cache_path)

        # Convert lists back to appropriate types if needed (parquet handles lists well)
        return QADataset(
            input_ids=df["input_ids"].tolist(),
            attention_mask=df["attention_mask"].tolist(),
            start_positions=df["start_position"].tolist(),
            end_positions=df["end_position"].tolist(),
            relevance_labels=df["relevance"].tolist(),
        )

    print("Processing training features from scratch...")

    # 2. Load Data
    # Merge train and val as per "Full-Data" strategy
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Debugging subsample
    if config.debug:
        df = df.head(config.debug_sample_size)

    # 3. Tokenization & Labeling
    # Pre-calculate answer end character index
    df["answer_text"] = df["answer_text"].astype(str)
    df["answer_end"] = df["answer_start"] + df["answer_text"].apply(len)

    # Tokenize
    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=config.max_len,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Temporary storage for classification
    features_by_sample = {}

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        row = df.iloc[sample_index]

        ans_start_char = row["answer_start"]
        ans_end_char = row["answer_end"]

        # Find the context start and end indices in the tokens
        # sequence_ids: 0 for question, 1 for context, None for special tokens
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Check if answer is contained in this window
        # offsets[token_start_index][0] is the start char of the first context token
        # offsets[token_end_index][1] is the end char of the last context token

        is_contained = (offsets[token_start_index][0] <= ans_start_char) and (
            offsets[token_end_index][1] >= ans_end_char
        )

        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sample_index": sample_index,
            "original_index": i,  # to track adjacency
        }

        if is_contained:
            # Map char positions to token positions
            # Move start pointer
            current_idx = token_start_index
            while (
                current_idx <= token_end_index
                and offsets[current_idx][0] <= ans_start_char
            ):
                current_idx += 1
            start_token_pos = current_idx - 1

            # Move end pointer
            current_idx = token_end_index
            while (
                current_idx >= token_start_index
                and offsets[current_idx][1] >= ans_end_char
            ):
                current_idx -= 1
            end_token_pos = current_idx + 1

            feature["start_position"] = start_token_pos
            feature["end_position"] = end_token_pos
            feature["relevance"] = 1.0
            feature["type"] = "positive"
        else:
            feature["start_position"] = 0  # CLS
            feature["end_position"] = 0  # CLS
            feature["relevance"] = 0.0
            feature["type"] = "negative"

        if sample_index not in features_by_sample:
            features_by_sample[sample_index] = []
        features_by_sample[sample_index].append(feature)

    # 4. Sampling Strategy (Hard Negative Mining)
    final_features = []

    for sample_idx, feats in features_by_sample.items():
        # Identify positives
        positives = [f for f in feats if f["type"] == "positive"]
        negatives = [f for f in feats if f["type"] == "negative"]

        # If no positive found (e.g. answer too long or weird truncation), skip sample or take all negatives?
        # Usually we skip to avoid noise, but let's keep negatives to learn "no answer"
        # However, for this task, every question HAS an answer in the context (extraction).
        # If we lost it, it's a data error. We will skip this sample to avoid confusing the model.
        if not positives:
            continue

        # Identify Boundary Negatives
        # Windows adjacent to a positive window in the sliding window sequence
        pos_indices = {f["original_index"] for f in positives}
        boundary_negatives = []
        random_negatives = []

        for neg in negatives:
            idx = neg["original_index"]
            # Check adjacency
            if (idx - 1 in pos_indices) or (idx + 1 in pos_indices):
                boundary_negatives.append(neg)
            else:
                random_negatives.append(neg)

        # Calculate quotas
        n_pos = len(positives)
        n_neg_target = n_pos * config.negative_positive_ratio

        # Select Negatives
        selected_negatives = []

        # Combine all negatives (Boundary + Random) and sample uniformly
        # We avoid prioritizing boundary negatives as they can introduce label noise (Cite solution_lesson_node_00038)
        all_negatives = boundary_negatives + random_negatives

        if len(all_negatives) > n_neg_target:
            selected_negatives.extend(random.sample(all_negatives, int(n_neg_target)))
        else:
            selected_negatives.extend(all_negatives)

        final_features.extend(positives)
        final_features.extend(selected_negatives)

    # 5. Save to Cache
    # Convert to DataFrame for easy Parquet saving
    data_dict = {
        "input_ids": [f["input_ids"] for f in final_features],
        "attention_mask": [f["attention_mask"] for f in final_features],
        "start_position": [f["start_position"] for f in final_features],
        "end_position": [f["end_position"] for f in final_features],
        "relevance": [f["relevance"] for f in final_features],
    }
    out_df = pd.DataFrame(data_dict)

    # Ensure directory exists
    os.makedirs(config.working_dir, exist_ok=True)
    out_df.to_parquet(cache_path)
    print(f"Saved {len(out_df)} features to {cache_path}")

    return QADataset(
        input_ids=data_dict["input_ids"],
        attention_mask=data_dict["attention_mask"],
        start_positions=data_dict["start_position"],
        end_positions=data_dict["end_position"],
        relevance_labels=data_dict["relevance"],
    )


def prepare_test_features(config, tokenizer):
    """
    Prepares test features using exhaustive sliding windows.
    Returns the dataset and a list of feature metadata for reconstruction.
    """
    df = pd.read_csv(config.test_path)

    if config.debug:
        df = df.head(config.debug_sample_size)

    # Tokenize
    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=config.max_len,
        stride=config.doc_stride,
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
        context_text = df.iloc[sample_index]["context"]

        # Mask offsets that are not part of the context
        # Set offsets to None for question tokens and special tokens
        final_offsets = []
        for k, seq_id in enumerate(sequence_ids):
            if seq_id != 1:
                final_offsets.append(None)
            else:
                final_offsets.append(offsets[k])

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "example_id": example_id,
                "offset_mapping": final_offsets,
                "context": context_text,
                "feature_index": i,
            }
        )

    # Create Dataset
    dataset = QADataset(
        input_ids=[f["input_ids"] for f in features],
        attention_mask=[f["attention_mask"] for f in features],
    )

    return dataset, features
