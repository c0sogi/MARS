import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles both training (with labels) and testing (without labels) modes.
    """

    def __init__(self, features, mode="train"):
        self.mode = mode
        self.input_ids = features["input_ids"]
        self.attention_mask = features["attention_mask"]

        if self.mode == "train":
            self.start_positions = features["start_positions"]
            self.end_positions = features["end_positions"]
            self.relevance_labels = features["relevance_labels"]
        else:
            # For inference, we track IDs to map back to original text
            self.example_ids = features["example_id"]
            # offset_mapping is stored in the dataframe/features but not usually converted to tensor
            # We access it via the index during inference post-processing

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "index": idx,  # Return index to map back to metadata (offsets, etc.)
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

        return item


def prepare_features(examples, tokenizer, mode="train"):
    """
    Tokenizes examples with sliding windows and generates labels.

    Args:
        examples (pd.DataFrame): DataFrame containing 'question', 'context', etc.
        tokenizer: Transformers tokenizer.
        mode (str): 'train' or 'test'.

    Returns:
        dict: Dictionary of features suitable for DataFrame construction.
    """

    # Tokenize with sliding windows
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = {
        "input_ids": [],
        "attention_mask": [],
        "offset_mapping": [],
        "example_id": [],
    }

    if mode == "train":
        features["start_positions"] = []
        features["end_positions"] = []
        features["relevance_labels"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        example_id = examples.iloc[sample_index]["id"]

        features["input_ids"].append(input_ids)
        features["attention_mask"].append(attention_mask)
        # Convert list of tuples to list of lists for Parquet compatibility
        features["offset_mapping"].append([list(o) for o in offsets])
        features["example_id"].append(example_id)

        if mode == "train":
            start_char = examples.iloc[sample_index]["answer_start"]
            answer_text = examples.iloc[sample_index]["answer_text"]
            end_char = start_char + len(answer_text)

            # Determine the context span within the token sequence
            # sequence_ids: None (special), 0 (question), None, 1 (context), None

            # Find start index of context
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # Find end index of context
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Check if the answer is fully contained in this window
            # offsets[token_start_index][0] is the start char of the first context token
            # offsets[token_end_index][1] is the end char of the last context token
            if (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):

                # Move token_start_index forward to the token containing start_char
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                start_position = token_start_index - 1

                # Move token_end_index backward to the token containing end_char
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_position = token_end_index + 1

                relevance = 1.0
            else:
                # Answer is not fully in this window
                start_position = 0  # Point to CLS
                end_position = 0  # Point to CLS
                relevance = 0.0

            features["start_positions"].append(start_position)
            features["end_positions"].append(end_position)
            features["relevance_labels"].append(relevance)

    return features


def get_processed_data(tokenizer, mode="train", load_cached_data=True):
    """
    Retrieves processed features from cache or processes raw data.

    Args:
        tokenizer: Transformers tokenizer.
        mode (str): 'train' or 'test'.
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        pd.DataFrame: Processed features.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_features.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from scratch...")

    if mode == "train":
        # Full-Data Strategy: Merge Train and Val metadata
        train_df = pd.read_csv(Config.TRAIN_META_PATH)
        val_df = pd.read_csv(Config.VAL_META_PATH)
        examples = pd.concat([train_df, val_df], ignore_index=True)

        if Config.DEBUG:
            examples = examples.head(Config.DEBUG_SAMPLE_SIZE)
    else:
        examples = pd.read_csv(Config.TEST_META_PATH)
        if Config.DEBUG:
            examples = examples.head(Config.DEBUG_SAMPLE_SIZE)

    features_dict = prepare_features(examples, tokenizer, mode=mode)
    df = pd.DataFrame(features_dict)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved {mode} features to {cache_path}")

    return df


def create_loaders(tokenizer, load_cached_data=True):
    """
    Generates DataLoaders for training and testing.
    Applies negative sampling to the training set.

    Args:
        tokenizer: Transformers tokenizer.
        load_cached_data (bool): Whether to use cached features.

    Returns:
        tuple: (train_loader, test_loader, test_features_df)
    """
    # Use the first seed for reproducible data sampling
    set_seed(Config.SEEDS[0])

    # --- Train Data ---
    train_features_df = get_processed_data(
        tokenizer, mode="train", load_cached_data=load_cached_data
    )

    # Negative Sampling Logic
    # Filter positives and negatives
    positives = train_features_df[train_features_df["relevance_labels"] == 1.0]
    negatives = train_features_df[train_features_df["relevance_labels"] == 0.0]

    n_pos = len(positives)
    # Calculate target number of negatives based on ratio
    n_neg_to_keep = int(n_pos * Config.NEG_TO_POS_RATIO)
    # Ensure we don't ask for more negatives than exist
    n_neg_to_keep = min(n_neg_to_keep, len(negatives))

    # Sample negatives
    if n_neg_to_keep > 0:
        negatives_sampled = negatives.sample(
            n=n_neg_to_keep, random_state=Config.SEEDS[0]
        )
    else:
        negatives_sampled = pd.DataFrame()

    # Combine and shuffle
    train_sampled = (
        pd.concat([positives, negatives_sampled])
        .sample(frac=1, random_state=Config.SEEDS[0])
        .reset_index(drop=True)
    )

    print(
        f"Training Data Stats: {len(positives)} Positives, {len(negatives_sampled)} Negatives. Total: {len(train_sampled)}"
    )

    # Create Dataset
    train_data = {col: train_sampled[col].tolist() for col in train_sampled.columns}
    train_dataset = QADataset(train_data, mode="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test Data ---
    test_features_df = get_processed_data(
        tokenizer, mode="test", load_cached_data=load_cached_data
    )

    # Use all windows for testing (no sampling)
    test_data = {
        col: test_features_df[col].tolist() for col in test_features_df.columns
    }
    test_dataset = QADataset(test_data, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, test_features_df
