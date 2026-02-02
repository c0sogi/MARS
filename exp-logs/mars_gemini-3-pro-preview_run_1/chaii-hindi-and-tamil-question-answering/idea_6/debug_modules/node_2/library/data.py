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
    Wraps processed features (input_ids, attention_mask, labels) into tensors.
    """

    def __init__(self, features):
        """
        Args:
            features (dict): Dictionary containing lists of features.
                             Keys: input_ids, attention_mask,
                                   start_positions (train), end_positions (train), relevance_labels (train)
        """
        self.features = features

    def __len__(self):
        return len(self.features["input_ids"])

    def __getitem__(self, idx):
        item = {}
        # Basic inputs
        item["input_ids"] = torch.tensor(
            self.features["input_ids"][idx], dtype=torch.long
        )
        item["attention_mask"] = torch.tensor(
            self.features["attention_mask"][idx], dtype=torch.long
        )

        # Training targets
        if "start_positions" in self.features:
            item["start_positions"] = torch.tensor(
                self.features["start_positions"][idx], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(
                self.features["end_positions"][idx], dtype=torch.long
            )
            # Relevance labels for BCEWithLogitsLoss (float)
            item["relevance_labels"] = torch.tensor(
                self.features["relevance_labels"][idx], dtype=torch.float
            )

        return item


def prepare_train_features(seed, load_cached_data=True):
    """
    Prepares training features with sliding windows and seeded negative sampling.

    Args:
        seed (int): Random seed for negative sampling.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        QADataset: The dataset ready for training.
    """
    # Set seed for reproducibility of sampling
    set_seed(seed)

    # Cache filename specific to the seed
    cache_path = os.path.join(Config.CACHE_DIR, f"train_features_seed_{seed}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training features from {cache_path}...")
        df_features = pd.read_parquet(cache_path)
        # Convert DataFrame back to dict of lists for Dataset
        features = df_features.to_dict(orient="list")
        return QADataset(features)

    print(f"Generating training features for seed {seed}...")

    # 2. Load and Merge Data
    # We use metadata files which are guaranteed to be correct splits,
    # but for full training we merge them back.
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Ensure correct types
    df["answer_text"] = df["answer_text"].astype(str)
    df["answer_start"] = df["answer_start"].astype(int)
    df["context"] = df["context"].astype(str)
    df["question"] = df["question"].astype(str)

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 4. Processing Loop
    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LEN,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflowing_tokens_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Lists to store positive and negative samples
    pos_input_ids = []
    pos_attention_mask = []
    pos_start_positions = []
    pos_end_positions = []
    pos_relevance = []

    neg_input_ids = []
    neg_attention_mask = []
    neg_start_positions = []
    neg_end_positions = []
    neg_relevance = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answer_text = df.iloc[sample_index]["answer_text"]
        start_char = df.iloc[sample_index]["answer_start"]
        end_char = start_char + len(answer_text)

        # Find context start and end tokens
        # sequence_ids: 0 for question, 1 for context (usually)
        # XLM-R: <s> Q </s> </s> C </s> -> None, 0, ..., None, None, 1, ..., None

        token_start_index = 0
        while (
            token_start_index < len(sequence_ids)
            and sequence_ids[token_start_index] != 1
        ):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If no context found (should not happen with proper truncation), treat as negative
        if token_start_index >= len(sequence_ids) or token_end_index < 0:
            neg_input_ids.append(input_ids)
            neg_attention_mask.append(attention_mask)
            neg_start_positions.append(0)
            neg_end_positions.append(0)
            neg_relevance.append(0.0)
            continue

        # Check if answer is fully contained in this window
        # offsets[token_start_index][0] is the start char of the first context token
        # offsets[token_end_index][1] is the end char of the last context token

        if not (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            # Answer not fully in window -> Negative sample
            neg_input_ids.append(input_ids)
            neg_attention_mask.append(attention_mask)
            neg_start_positions.append(0)
            neg_end_positions.append(0)
            neg_relevance.append(0.0)
        else:
            # Answer is in window -> Positive sample
            # Move start index forward to find the token starting at or after start_char
            current_idx = token_start_index
            while (
                current_idx <= token_end_index and offsets[current_idx][0] <= start_char
            ):
                current_idx += 1
            start_token = current_idx - 1

            # Move end index backward to find the token ending at or before end_char
            current_idx = token_end_index
            while (
                current_idx >= token_start_index and offsets[current_idx][1] >= end_char
            ):
                current_idx -= 1
            end_token = current_idx + 1

            pos_input_ids.append(input_ids)
            pos_attention_mask.append(attention_mask)
            pos_start_positions.append(start_token)
            pos_end_positions.append(end_token)
            pos_relevance.append(1.0)

    # 5. Negative Sampling
    n_pos = len(pos_input_ids)
    if n_pos > 0:
        target_neg = int(n_pos * Config.NEGATIVE_POSITIVE_RATIO)
        n_neg = min(len(neg_input_ids), target_neg)
    else:
        # Fallback if no positives found (unlikely)
        n_neg = len(neg_input_ids)

    # Shuffle negatives deterministically
    indices = np.arange(len(neg_input_ids))
    np.random.shuffle(indices)
    selected_indices = indices[:n_neg]

    final_input_ids = pos_input_ids + [neg_input_ids[i] for i in selected_indices]
    final_attention_mask = pos_attention_mask + [
        neg_attention_mask[i] for i in selected_indices
    ]
    final_start_positions = pos_start_positions + [
        neg_start_positions[i] for i in selected_indices
    ]
    final_end_positions = pos_end_positions + [
        neg_end_positions[i] for i in selected_indices
    ]
    final_relevance = pos_relevance + [neg_relevance[i] for i in selected_indices]

    # Shuffle the combined dataset
    combined_indices = np.arange(len(final_input_ids))
    np.random.shuffle(combined_indices)

    final_features = {
        "input_ids": [final_input_ids[i] for i in combined_indices],
        "attention_mask": [final_attention_mask[i] for i in combined_indices],
        "start_positions": [final_start_positions[i] for i in combined_indices],
        "end_positions": [final_end_positions[i] for i in combined_indices],
        "relevance_labels": [final_relevance[i] for i in combined_indices],
    }

    # 6. Save to Cache
    df_out = pd.DataFrame(final_features)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_out.to_parquet(cache_path)
    print(f"Saved {len(df_out)} training samples to {cache_path}")

    return QADataset(final_features)


def prepare_test_features(load_cached_data=True):
    """
    Prepares test features with exhaustive sliding windows.

    Args:
        load_cached_data (bool): Whether to load from cache.

    Returns:
        tuple: (QADataset, pd.DataFrame)
               The dataset for the model, and a DataFrame with metadata for reconstruction.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "test_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test features from {cache_path}...")
        df_features = pd.read_parquet(cache_path)

        # Separate features for Dataset and Metadata
        dataset_cols = ["input_ids", "attention_mask"]
        dataset_dict = {k: df_features[k].tolist() for k in dataset_cols}

        return QADataset(dataset_dict), df_features

    print("Generating test features...")
    test_df = pd.read_csv(Config.TEST_CSV)

    # Ensure types
    test_df["context"] = test_df["context"].astype(str)
    test_df["question"] = test_df["question"].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    tokenized_examples = tokenizer(
        test_df["question"].tolist(),
        test_df["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LEN,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflowing_tokens_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Collect data
    input_ids_list = []
    attention_mask_list = []
    example_ids = []
    offset_mappings = []
    sequence_ids_list = []
    contexts = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # Convert sequence_ids to list, handle None
        seq_ids = tokenized_examples.sequence_ids(i)
        # Replace None with -1 for serialization safety
        seq_ids_safe = [x if x is not None else -1 for x in seq_ids]

        sample_index = sample_mapping[i]
        example_id = test_df.iloc[sample_index]["id"]
        context_text = test_df.iloc[sample_index]["context"]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        example_ids.append(example_id)
        offset_mappings.append(offsets)  # List of tuples
        sequence_ids_list.append(seq_ids_safe)
        contexts.append(context_text)

    # Create DataFrame
    df_features = pd.DataFrame(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "example_id": example_ids,
            "offset_mapping": offset_mappings,
            "sequence_ids": sequence_ids_list,
            "context": contexts,
        }
    )

    # Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path)
    print(f"Saved {len(df_features)} test windows to {cache_path}")

    # Return Dataset and DataFrame
    dataset_dict = {"input_ids": input_ids_list, "attention_mask": attention_mask_list}

    return QADataset(dataset_dict), df_features
