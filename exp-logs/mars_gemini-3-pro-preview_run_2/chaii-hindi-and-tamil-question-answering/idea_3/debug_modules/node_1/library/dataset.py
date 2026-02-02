import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        # Add labels if they exist
        if "start_positions" in row and not np.isnan(row["start_positions"]):
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)

        # Add inference/validation metadata if available
        if "offset_mapping" in row:
            # offset_mapping is stored as a list of lists/tuples in parquet
            item["offset_mapping"] = torch.tensor(
                row["offset_mapping"], dtype=torch.long
            )

        if "sequence_ids" in row:
            # Replace None with -1 for tensor conversion
            seq_ids = row["sequence_ids"]
            seq_ids = [-1 if x is None else x for x in seq_ids]
            item["sequence_ids"] = torch.tensor(seq_ids, dtype=torch.long)

        if "example_id" in row:
            item["example_id"] = str(row["example_id"])

        return item


def process_data(df, tokenizer, has_labels=True):
    """
    Processes the dataframe using the tokenizer with sliding window strategy.
    """
    # Clean text
    df["question"] = df["question"].astype(str).str.strip()
    df["context"] = df["context"].astype(str).str.strip()

    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Prepare output storage
    final_features = {
        "input_ids": tokenized_examples["input_ids"],
        "attention_mask": tokenized_examples["attention_mask"],
        "offset_mapping": [],
        "sequence_ids": [],
        "example_id": [],
    }

    if has_labels:
        final_features["start_positions"] = []
        final_features["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]

        # Get sequence IDs (0 for question, 1 for context, None for special)
        # We can't use .sequence_ids() directly on the batch object easily in loop without index
        # But tokenized_examples is a BatchEncoding, so we can access it if we used the object.
        # However, we popped keys. Let's reconstruct or use the fast way.
        # Fast tokenizer returns sequence_ids method on the object but we have lists now.
        # We need to re-derive or rely on the fact that we can get it from the tokenizer result object
        # BEFORE popping.
        # Re-approach: Don't pop immediately or use the batch encoding object methods.

        # Correct approach: Use the `sequence_ids` method of the batch encoding for the specific index
        seq_ids = tokenized_examples.sequence_ids(i)
        final_features["sequence_ids"].append(seq_ids)

        # Convert offsets to list of lists for compatibility
        final_features["offset_mapping"].append([list(o) for o in offsets])

        # Map back to original example
        sample_index = sample_mapping[i]
        example_id = df.iloc[sample_index]["id"]
        final_features["example_id"].append(example_id)

        if has_labels:
            answer_text = df.iloc[sample_index]["answer_text"]
            start_char = df.iloc[sample_index]["answer_start"]
            end_char = start_char + len(answer_text)

            # Find the context start and end in the token sequence
            # seq_ids: None (special), 0 (question), None, 1 (context), None

            # Find index of first '1' and last '1'
            token_start_index = 0
            while seq_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while seq_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (sliding window)
            # offsets[token_start_index][0] is the start char of the first context token
            # offsets[token_end_index][1] is the end char of the last context token

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                # Answer is not fully inside the context window
                final_features["start_positions"].append(0)
                final_features["end_positions"].append(0)
            else:
                # Move token_start_index and token_end_index to the answer boundaries
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                final_features["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                final_features["end_positions"].append(token_end_index + 1)

    return pd.DataFrame(final_features)


def get_data(tokenizer, split="train", load_cached_data=True):
    """
    Loads data for the specified split.
    Uses caching to avoid re-processing.

    Args:
        tokenizer: HuggingFace tokenizer.
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: Processed features.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache filename
    debug_suffix = "_debug" if Config.DEBUG else ""
    cache_filename = f"cached_{split}{debug_suffix}_features.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} features from scratch...")

    # Identify source file
    if split == "train":
        file_path = Config.TRAIN_CSV
        has_labels = True
    elif split == "val":
        file_path = Config.VAL_CSV
        has_labels = True
    elif split == "test":
        file_path = Config.TEST_CSV
        has_labels = False
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load raw data
    df = pd.read_csv(file_path)

    # Handle DEBUG mode
    if Config.DEBUG:
        print(
            f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows from {split} data."
        )
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Process data
    processed_df = process_data(df, tokenizer, has_labels=has_labels)

    # Save to cache
    print(f"Saving {split} features to cache: {cache_path}")
    processed_df.to_parquet(cache_path, index=False)

    return processed_df
