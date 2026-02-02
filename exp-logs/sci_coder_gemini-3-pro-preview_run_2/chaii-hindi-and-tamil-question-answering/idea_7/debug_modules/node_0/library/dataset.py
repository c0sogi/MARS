import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


def prepare_train_features(examples, tokenizer):
    """
    Tokenizes training data with sliding window and generates labels.
    """
    # Clean whitespace
    examples["question"] = examples["question"].str.strip()

    # Tokenize
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LEN,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    # Iterate over each window (feature) generated
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # Map back to original example
        sample_index = sample_mapping[i]
        answers = examples.iloc[sample_index]
        start_char = answers["answer_start"]
        end_char = start_char + len(answers["answer_text"])

        # Sequence IDs: None (special), 0 (question), 1 (context)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context in the input_ids
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Check if the answer is fully inside the context of this window
        # offsets[context_start][0] is the start char of the first context token
        # offsets[context_end][1] is the end char of the last context token
        if not (
            offsets[context_start][0] <= start_char
            and offsets[context_end][1] >= end_char
        ):
            # Answer is not fully in this window
            start_position = 0
            end_position = 0
            answerable_label = 0.0
        else:
            # Answer is in this window
            # Move the token_start_index to the start of the answer
            token_start_index = context_start
            while (
                token_start_index <= context_end
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            start_position = token_start_index - 1

            # Move the token_end_index to the end of the answer
            token_end_index = context_end
            while (
                token_end_index >= context_start
                and offsets[token_end_index][1] >= end_char
            ):
                token_end_index -= 1
            end_position = token_end_index + 1

            answerable_label = 1.0

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "answerable_label": answerable_label,
                "example_id": str(answers["id"]),
            }
        )

    return pd.DataFrame(features)


def prepare_test_features(examples, tokenizer):
    """
    Tokenizes test/validation data with sliding window for inference.
    Retains offset mapping for post-processing.
    """
    examples["question"] = examples["question"].str.strip()

    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LEN,
        stride=Config.DOC_STRIDE,
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
        example_id = examples.iloc[sample_index]["id"]
        context_text = examples.iloc[sample_index]["context"]

        # Set offsets for non-context tokens to None to avoid selecting them
        # We keep the raw offsets list but will process it later or store it as is
        # For dataframe storage, lists are fine.

        # Identify context tokens for later use (optional but good for debugging)
        # We store the raw offset_mapping.

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,
                "example_id": str(example_id),
                "sequence_ids": sequence_ids,  # Useful for post-processing to identify context
                "context": context_text,  # Store context for string extraction
            }
        )

    return pd.DataFrame(features)


def get_data(tokenizer=None, load_cached_data=True, split="train"):
    """
    Main function to load and process data.
    Handles caching logic.

    Args:
        tokenizer: Pre-initialized tokenizer (required if not loading cache).
        load_cached_data: Boolean to attempt loading from parquet.
        split: 'train', 'val', or 'test'.
    """
    # Construct cache path
    prefix = "debug_" if Config.DEBUG else ""
    cache_filename = f"{prefix}cached_{split}_features.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    print(f"Processing {split} data from metadata...")

    if split == "train":
        meta_path = Config.TRAIN_META
        process_func = prepare_train_features
    elif split == "val":
        meta_path = Config.VAL_META
        # For validation, we might want targets for eval, but usually we treat val like test
        # for prediction and then compute metrics externally.
        # However, to compute loss during training, we need labels.
        # Let's check if we need to run validation loop with loss or just metrics.
        # Usually standard training loops compute val loss. So we use prepare_train_features logic
        # if we want labels.
        # But for the final metric calculation (Jaccard), we need the raw text inference.
        # We will generate features with labels for loss calculation.
        process_func = prepare_train_features
    elif split == "test":
        meta_path = Config.TEST_META
        process_func = prepare_test_features
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_raw = pd.read_csv(meta_path)

    if Config.DEBUG:
        df_raw = df_raw.head(Config.DEBUG_SAMPLE_SIZE)

    df_processed = process_func(df_raw, tokenizer)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    # Parquet handles lists/arrays well
    df_processed.to_parquet(cache_path, index=False)
    print(f"Saved {len(df_processed)} features to {cache_path}")

    return df_processed


class QADataset(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Common inputs
        # Ensure they are numpy arrays or lists before converting to tensor
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)

        if self.is_test:
            # For inference, we mainly need inputs.
            # We return the index or ID to map back to the dataframe row in the loop
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "index": torch.tensor(idx, dtype=torch.long),
            }
        else:
            # Training/Validation with labels
            start_positions = torch.tensor(row["start_positions"], dtype=torch.long)
            end_positions = torch.tensor(row["end_positions"], dtype=torch.long)
            answerable_label = torch.tensor(row["answerable_label"], dtype=torch.float)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "answerable_label": answerable_label,
            }
