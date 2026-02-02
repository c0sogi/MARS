import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Wraps the processed features (input_ids, attention_mask, labels).
    """

    def __init__(self, features_df):
        self.features = features_df.reset_index(drop=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        row = self.features.iloc[idx]

        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        # Add labels if they exist (for training)
        if "start_positions" in row and not pd.isna(row["start_positions"]):
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)

        return item


def prepare_features(examples, tokenizer, is_training=True):
    """
    Tokenizes examples with sliding window and generates labels for QA.
    """
    # Clean whitespace in questions
    questions = [q.strip() for q in examples["question"]]

    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        questions,
        examples["context"].tolist(),
        truncation="only_second",  # Truncate context, not question
        max_length=Config.max_length,
        stride=Config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialize lists for new features
    new_features = {
        "input_ids": tokenized_examples["input_ids"],
        "attention_mask": tokenized_examples["attention_mask"],
        "example_id": [],
        "offset_mapping": offset_mapping,
    }

    if is_training:
        new_features["start_positions"] = []
        new_features["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is context and what is question)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        new_features["example_id"].append(examples.iloc[sample_index]["id"])

        if is_training:
            ans_text = examples.iloc[sample_index]["answer_text"]
            start_char = examples.iloc[sample_index]["answer_start"]

            # Handle edge cases where answer might be NaN (though dataset analysis says 0 missing)
            if pd.isna(ans_text):
                # If no answer, point to CLS
                new_features["start_positions"].append(cls_index)
                new_features["end_positions"].append(cls_index)
                continue

            end_char = start_char + len(str(ans_text))

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
                new_features["start_positions"].append(cls_index)
                new_features["end_positions"].append(cls_index)
            else:
                # Move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go more granular, but this is the standard HF approach.
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                new_features["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                new_features["end_positions"].append(token_end_index + 1)

    return pd.DataFrame(new_features)


def load_or_create_features(
    tokenizer, split, raw_data_path, cache_path, load_cached_data=True
):
    """
    Manages caching of processed features.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path, engine="pyarrow")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data from {raw_data_path}...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    raw_df = pd.read_csv(raw_data_path)

    # Debugging subsample
    if Config.debug:
        print(f"Debug mode: Subsampling {Config.debug_sample_size} rows.")
        raw_df = raw_df.head(Config.debug_sample_size)

    # Determine if training mode (has answers)
    is_training = "answer_text" in raw_df.columns and split != "test"

    features_df = prepare_features(raw_df, tokenizer, is_training=is_training)

    # 3. Save to cache
    print(f"Saving processed features to {cache_path}...")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        features_df.to_parquet(cache_path, engine="pyarrow")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}. Error: {e}")

    return features_df


def get_train_dataset(tokenizer, load_cached_data=True):
    """
    Returns the training dataset.
    """
    cache_path = Config.cached_train_features_path
    features_df = load_or_create_features(
        tokenizer, "train", Config.train_meta_path, cache_path, load_cached_data
    )
    return QADataset(features_df)


def get_val_dataset(tokenizer, load_cached_data=True):
    """
    Returns the validation dataset and the raw dataframe (for metric calculation).
    """
    # Construct a cache path for validation (not explicitly in Config, so we derive it)
    cache_path = os.path.join(Config.output_dir, "cached_val_features.parquet")

    features_df = load_or_create_features(
        tokenizer, "val", Config.val_meta_path, cache_path, load_cached_data
    )

    # We also need the raw data to compare predictions against ground truth text
    raw_df = pd.read_csv(Config.val_meta_path)
    if Config.debug:
        raw_df = raw_df.head(Config.debug_sample_size)

    return QADataset(features_df), raw_df, features_df


def get_test_dataset(tokenizer, load_cached_data=True):
    """
    Returns the test dataset and the raw dataframe (for submission generation).
    """
    cache_path = Config.cached_test_features_path
    features_df = load_or_create_features(
        tokenizer, "test", Config.test_meta_path, cache_path, load_cached_data
    )

    raw_df = pd.read_csv(Config.test_meta_path)
    if Config.debug:
        raw_df = raw_df.head(Config.debug_sample_size)

    return QADataset(features_df), raw_df, features_df
