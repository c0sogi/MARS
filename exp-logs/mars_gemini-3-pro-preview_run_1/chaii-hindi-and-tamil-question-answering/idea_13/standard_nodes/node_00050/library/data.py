import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import json
import ast

from library.config import Config
from library.utils import seed_everything


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    """

    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Parse input_ids and attention_mask from string/list if necessary
        # (Parquet might load them as arrays, but just to be safe)
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]

        # Convert to tensors
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if not self.is_test:
            item["start_positions"] = torch.tensor(
                row["start_position"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_position"], dtype=torch.long)
            item["relevance"] = torch.tensor(row["relevance"], dtype=torch.float)
        else:
            # For test, we might need example_id and offset_mapping for post-processing
            # We don't return them as tensors, but they are accessible in the dataframe
            pass

        return item


def process_data_to_features(examples, tokenizer, is_test=False):
    """
    Converts raw examples to tokenized features with sliding windows.
    """

    # Tokenize
    # questions are the first sequence, contexts are the second
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.max_length,
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

        # Map back to original example
        sample_index = sample_mapping[i]

        # Base feature dict
        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": json.dumps(offsets),  # Serialize for storage
            "example_id": examples.iloc[sample_index]["id"],
            "sequence_ids": json.dumps(tokenized_examples.sequence_ids(i)),  # Serialize
        }

        if not is_test:
            # Label generation
            answer_text = examples.iloc[sample_index]["answer_text"]
            start_char = examples.iloc[sample_index]["answer_start"]
            end_char = start_char + len(answer_text)

            sequence_ids = tokenized_examples.sequence_ids(i)

            # Find the context start and end in tokens
            # sequence_ids: None (special), 0 (question), 1 (context)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            token_start_index = idx

            while sequence_ids[idx] == 1:
                idx += 1
            token_end_index = idx - 1

            # Check if answer is contained in this window
            # offset_mapping[token_index] returns (char_start, char_end)

            # If the answer is not fully inside the context in this window, label as 0,0
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                feature["start_position"] = 0
                feature["end_position"] = 0
                feature["relevance"] = 0
            else:
                # Move token_start_index and token_end_index to the answer start/end
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                feature["start_position"] = token_start_index - 1

                while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                feature["end_position"] = token_end_index + 1

                feature["relevance"] = 1

        features.append(feature)

    return pd.DataFrame(features)


def prepare_train_features(tokenizer, load_cached_data=True):
    """
    Prepares training features with caching and negative sampling.
    """
    cache_path = Config.train_features_file

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached train features from {cache_path}")
        df = pd.read_parquet(cache_path)
        # Deserialize lists
        # Note: Parquet handles lists of primitives well, but we serialized offset_mapping/sequence_ids just in case
        # For input_ids/attention_mask, pandas read_parquet usually recovers them as numpy arrays or lists
        return df

    print("Processing train features from scratch...")

    # Load Metadata
    train_df = pd.read_csv(Config.train_meta_path)
    val_df = pd.read_csv(Config.val_meta_path)

    if Config.use_full_train_data:
        full_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        full_df = train_df

    # Process
    features_df = process_data_to_features(full_df, tokenizer, is_test=False)

    # Negative Sampling
    positives = features_df[features_df["relevance"] == 1]
    negatives = features_df[features_df["relevance"] == 0]

    n_pos = len(positives)
    n_neg_keep = int(n_pos * Config.negative_positive_ratio)

    # Deterministic sampling
    if len(negatives) > n_neg_keep:
        negatives = negatives.sample(n=n_neg_keep, random_state=Config.seed)

    # Combine and Shuffle
    final_df = (
        pd.concat([positives, negatives])
        .sample(frac=1, random_state=Config.seed)
        .reset_index(drop=True)
    )

    print(
        f"Train Features: {len(final_df)} samples ({len(positives)} pos, {len(negatives)} neg)"
    )

    # Save to Cache
    # Parquet requires consistent types. Lists are supported.
    # We ensure input_ids and attention_mask are lists
    final_df["input_ids"] = final_df["input_ids"].apply(list)
    final_df["attention_mask"] = final_df["attention_mask"].apply(list)

    final_df.to_parquet(cache_path, index=False)

    return final_df


def prepare_test_features(tokenizer, load_cached_data=True):
    """
    Prepares test features with caching (exhaustive sliding windows).
    """
    cache_path = Config.test_features_file

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test features from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    print("Processing test features from scratch...")

    # Load Metadata
    test_df = pd.read_csv(Config.test_meta_path)

    # Process
    features_df = process_data_to_features(test_df, tokenizer, is_test=True)

    print(
        f"Test Features: {len(features_df)} samples generated from {len(test_df)} docs."
    )

    # Save to Cache
    features_df["input_ids"] = features_df["input_ids"].apply(list)
    features_df["attention_mask"] = features_df["attention_mask"].apply(list)

    features_df.to_parquet(cache_path, index=False)

    return features_df


def get_tokenizer():
    return AutoTokenizer.from_pretrained(Config.model_name)


def get_train_dataloader(tokenizer, load_cached_data=True):
    seed_everything(Config.seed)
    df = prepare_train_features(tokenizer, load_cached_data=load_cached_data)
    dataset = QADataset(df, is_test=False)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )


def get_test_dataloader(tokenizer, load_cached_data=True):
    # No seed needed for test processing, but good practice
    df = prepare_test_features(tokenizer, load_cached_data=load_cached_data)
    dataset = QADataset(df, is_test=True)

    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=Config.eval_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        ),
        df,
    )
