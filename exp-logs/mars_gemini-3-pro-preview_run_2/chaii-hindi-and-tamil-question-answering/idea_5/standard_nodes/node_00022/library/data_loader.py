import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Apply seed
seed_everything(Config.SEED)


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles both training (with targets) and testing (with metadata) modes.
    """

    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Convert lists/arrays to tensors
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(row["token_type_ids"], dtype=torch.long),
            "example_id": row["example_id"],
        }

        if self.mode == "train":
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)
        else:
            # For inference, we need offset_mapping to map tokens back to chars
            # offset_mapping is a list of tuples (start, end), convert to tensor
            item["offset_mapping"] = torch.tensor(
                row["offset_mapping"], dtype=torch.long
            )

        return item


def prepare_train_features(examples, tokenizer):
    """
    Tokenizes training examples with sliding window and maps character answer spans to token indices.
    """
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=False,  # DistilBERT doesn't need these for model
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["example_id"] = []
    tokenized_examples["token_type_ids"] = []  # Manually creating for post-processing

    for i, offsets in enumerate(offset_mapping):
        # Generate token_type_ids manually for masking: 1 for context, 0 otherwise
        seq_ids = tokenized_examples.sequence_ids(i)
        # sequence_ids: None (special), 0 (question), 1 (context)
        # We map 1 -> 1, others -> 0
        token_type_ids = [1 if s == 1 else 0 for s in seq_ids]
        tokenized_examples["token_type_ids"].append(token_type_ids)

        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]

        # Store example_id for reference if needed (though not strictly used in train loop usually)
        tokenized_examples["example_id"].append(examples.iloc[sample_index]["id"])

        answers = examples.iloc[sample_index]["answer_text"]
        start_char = examples.iloc[sample_index]["answer_start"]

        # If answer is missing or empty (edge case), label as CLS
        if pd.isna(answers) or pd.isna(start_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        end_char = start_char + len(answers)

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
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go more granular, but this is the standard approach.
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return pd.DataFrame(
        {
            "input_ids": tokenized_examples["input_ids"],
            "attention_mask": tokenized_examples["attention_mask"],
            "token_type_ids": tokenized_examples["token_type_ids"],
            "start_positions": tokenized_examples["start_positions"],
            "end_positions": tokenized_examples["end_positions"],
            "example_id": tokenized_examples["example_id"],
        }
    )


def prepare_test_features(examples, tokenizer):
    """
    Tokenizes test examples with sliding window.
    Retains offset_mapping and example_id for post-processing predictions.
    """
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We need to keep the example_id for each feature to aggregate predictions later
    tokenized_examples["example_id"] = []
    tokenized_examples["token_type_ids"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples.iloc[sample_index]["id"])

        # Generate token_type_ids manually for masking
        seq_ids = tokenized_examples.sequence_ids(i)
        token_type_ids = [1 if s == 1 else 0 for s in seq_ids]
        tokenized_examples["token_type_ids"].append(token_type_ids)

    return pd.DataFrame(
        {
            "input_ids": tokenized_examples["input_ids"],
            "attention_mask": tokenized_examples["attention_mask"],
            "token_type_ids": tokenized_examples["token_type_ids"],
            "offset_mapping": tokenized_examples["offset_mapping"],
            "example_id": tokenized_examples["example_id"],
        }
    )


def get_processed_data(df, tokenizer, split="train", load_cached_data=True):
    """
    Handles caching logic.
    If cache exists and is requested, loads it.
    Otherwise, processes data and saves to cache.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} features from {cache_path}...")
        try:
            processed_df = pd.read_parquet(cache_path)
            print(f"Successfully loaded {len(processed_df)} features.")
            return processed_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {split} features...")
    if split == "test":
        processed_df = prepare_test_features(df, tokenizer)
    else:
        # train or val
        processed_df = prepare_train_features(df, tokenizer)

    print(f"Saving {len(processed_df)} features to {cache_path}...")
    # Ensure columns usually containing lists are compatible with parquet
    # PyArrow handles lists of ints/floats well.
    processed_df.to_parquet(cache_path, index=False)

    return processed_df
