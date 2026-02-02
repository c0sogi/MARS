import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles both training (with labels) and inference (without labels) modes.
    """

    def __init__(self, data, is_train=True):
        self.data = data
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Common features
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

        if self.is_train:
            # Training labels
            start_labels = torch.tensor(item["start_labels"], dtype=torch.long)
            end_labels = torch.tensor(item["end_labels"], dtype=torch.long)
            relevance_labels = torch.tensor(item["relevance_labels"], dtype=torch.float)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_labels": start_labels,
                "end_labels": end_labels,
                "relevance_labels": relevance_labels,
            }
        else:
            # Inference data
            # We return offset_mapping as a tensor for easier batching,
            # and example_index to link back to the original dataframe if needed.
            offset_mapping = torch.tensor(item["offset_mapping"], dtype=torch.long)

            # We return the example_id as a string (handled by collate_fn usually)
            # or we can rely on the caller to map indices.
            # Here we return the index in the original test dataframe if available,
            # but since we exploded the dataframe into windows, we pass the unique ID string.
            # Note: Default collate_fn fails with strings, so we wrap it or expect custom loop.
            # To be safe for standard loaders, we won't return strings in the tensor dict
            # if we expect standard batching. However, for inference loops, we often need it.
            # We will return the row index of the original dataframe (example_idx) if present.

            example_idx = torch.tensor(item.get("example_idx", -1), dtype=torch.long)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offset_mapping,
                "example_idx": example_idx,
            }


def process_examples(df, tokenizer, is_train=True, max_length=384, doc_stride=128):
    """
    Tokenizes examples with sliding window and generates labels.
    """
    features = []

    # Iterate row by row to handle logic explicitly
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        context = str(row["context"]).strip()

        # Tokenize with sliding window
        tokenized = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            attention_mask = tokenized["attention_mask"][i]
            sequence_ids = tokenized.sequence_ids(i)

            # Base feature dict
            feature = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if is_train:
                answer_text = str(row["answer_text"])
                start_char = row["answer_start"]
                end_char = start_char + len(answer_text)

                # Find the context start and end indices in the token list
                # sequence_ids: 0 for question, 1 for context, None for special tokens
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is fully inside the current window
                # offsets[token_start_index][0] is the start char of the first context token
                # offsets[token_end_index][1] is the end char of the last context token

                context_start_char = offsets[token_start_index][0]
                context_end_char = offsets[token_end_index][1]

                if not (
                    context_start_char <= start_char and end_char <= context_end_char
                ):
                    # Answer not fully in this window -> Negative Sample
                    feature["start_labels"] = 0
                    feature["end_labels"] = 0
                    feature["relevance_labels"] = 0.0
                    feature["is_positive"] = False
                else:
                    # Answer is in this window -> Positive Sample
                    # Map char index to token index

                    # Move token_start_index forward to the start of the answer
                    current_idx = token_start_index
                    while (
                        current_idx <= token_end_index
                        and offsets[current_idx][0] <= start_char
                    ):
                        current_idx += 1
                    start_token = current_idx - 1

                    # Move token_end_index backward to the end of the answer
                    current_idx = token_end_index
                    while (
                        current_idx >= token_start_index
                        and offsets[current_idx][1] >= end_char
                    ):
                        current_idx -= 1
                    end_token = current_idx + 1

                    feature["start_labels"] = start_token
                    feature["end_labels"] = end_token
                    feature["relevance_labels"] = 1.0
                    feature["is_positive"] = True
            else:
                # Inference specific metadata
                feature["offset_mapping"] = offsets
                # We store the index of the row in the original dataframe
                # This allows us to retrieve the example_id string later
                feature["example_idx"] = idx

            features.append(feature)

    return features


def prepare_data(load_cached_data=True):
    """
    Main function to load, process, cache, and return datasets.
    """
    seed_everything(42)  # Ensure deterministic processing

    # Paths for cache
    train_cache_path = os.path.join(Config.CACHE_DIR, "train_features.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_features.parquet")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print("Loading cached features from parquet...")
        train_df = pd.read_parquet(train_cache_path)
        test_df = pd.read_parquet(test_cache_path)

        # Convert back to list of dicts for Dataset
        train_features = train_df.to_dict("records")
        test_features = test_df.to_dict("records")

        # Reconstruct datasets
        train_dataset = QADataset(train_features, is_train=True)
        test_dataset = QADataset(test_features, is_train=False)

        return train_dataset, test_dataset

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    if Config.DEBUG:
        train_meta = train_meta.head(50)
        val_meta = val_meta.head(20)
        test_meta = test_meta.head(20)

    # Merge Train and Val if configured
    if Config.MERGE_TRAIN_VAL:
        print("Merging Train and Validation sets...")
        train_df_raw = pd.concat([train_meta, val_meta], ignore_index=True)
    else:
        train_df_raw = train_meta

    # Process Train
    print("Tokenizing and labeling training data...")
    raw_train_features = process_examples(
        train_df_raw,
        tokenizer,
        is_train=True,
        max_length=Config.MAX_LENGTH,
        doc_stride=Config.DOC_STRIDE,
    )

    # Negative Sampling Strategy
    positives = [f for f in raw_train_features if f["is_positive"]]
    negatives = [f for f in raw_train_features if not f["is_positive"]]

    n_pos = len(positives)
    n_neg_target = int(n_pos * Config.NEGATIVE_RATIO)

    # Deterministic sampling of negatives
    # We sort first to ensure stability before shuffle, though input order should be stable
    import random

    random.seed(42)
    random.shuffle(negatives)

    selected_negatives = negatives[:n_neg_target]

    train_features = positives + selected_negatives
    random.shuffle(train_features)

    print(
        f"Training Features: {len(train_features)} (Pos: {len(positives)}, Neg: {len(selected_negatives)})"
    )

    # Process Test
    print("Tokenizing test data...")
    test_features = process_examples(
        test_meta,
        tokenizer,
        is_train=False,
        max_length=Config.MAX_LENGTH,
        doc_stride=Config.DOC_STRIDE,
    )

    # 3. Save to Cache
    # Convert to DataFrame for Parquet saving
    # Note: Parquet handles lists in columns well
    train_feat_df = pd.DataFrame(train_features)
    # Drop the temporary 'is_positive' column if desired, or keep for analysis
    if "is_positive" in train_feat_df.columns:
        train_feat_df = train_feat_df.drop(columns=["is_positive"])

    test_feat_df = pd.DataFrame(test_features)

    print(f"Saving features to {Config.CACHE_DIR}...")
    train_feat_df.to_parquet(train_cache_path, index=False)
    test_feat_df.to_parquet(test_cache_path, index=False)

    # 4. Return Datasets
    train_dataset = QADataset(train_features, is_train=True)
    test_dataset = QADataset(test_features, is_train=False)

    return train_dataset, test_dataset
