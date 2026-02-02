import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Any, Optional

from library.config import Config


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Handles input tensors and metadata required for training and inference.
    """

    def __init__(self, data: pd.DataFrame, is_test: bool = False):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Convert list/array columns to tensors
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            # Metadata passed as raw values
            "offset_mapping": row["offset_mapping"],
            "example_id": row["example_id"],
            "context": row["context"],
        }

        if not self.is_test:
            # Labels: O=0, B-ANS=1, I-ANS=2, Ignore=-100
            item["labels"] = torch.tensor(row["labels"], dtype=torch.long)

        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle mixed types (tensors and metadata).
    Bundles metadata into lists while stacking tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Collect metadata as lists
    offset_mapping = [item["offset_mapping"] for item in batch]
    example_ids = [item["example_id"] for item in batch]
    contexts = [item["context"] for item in batch]

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
        "example_ids": example_ids,
        "contexts": contexts,
    }

    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        batch_out["labels"] = labels

    return batch_out


def prepare_features(
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
    load_cached_data: bool = True,
) -> pd.DataFrame:
    """
    Prepares features for the QA task using sliding window and soft overlap labeling.
    Handles caching to disk using Parquet.

    Args:
        config: Configuration object.
        tokenizer: HuggingFace tokenizer.
        split: 'train', 'val', or 'test'.
        load_cached_data: Whether to load from cache if available.

    Returns:
        pd.DataFrame containing processed features.
    """
    # 1. Determine Cache Path
    cache_filename = f"{split}_features.parquet"
    cache_path = os.path.join(config.cache_dir, cache_filename)

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features for '{split}' from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Preparing features for '{split}' split...")

    # 3. Load Metadata
    meta_path = os.path.join(config.metadata_dir, f"{split}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    examples = pd.read_csv(meta_path)
    # Ensure string types
    examples["question"] = examples["question"].fillna("").astype(str)
    examples["context"] = examples["context"].fillna("").astype(str)

    features = []

    # 4. Processing Loop
    # We iterate row by row to handle the sliding window and labeling logic precisely
    for _, row in examples.iterrows():
        question = row["question"]
        context = row["context"]
        example_id = row["id"]

        # Tokenize with sliding window
        # XLM-R structure: <s> Question </s> </s> Context </s>
        tokenized = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=config.max_length,
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=False,  # XLM-R relies on sequence_ids
        )

        offset_mappings = tokenized.pop("offset_mapping")
        input_ids_list = tokenized["input_ids"]
        attention_mask_list = tokenized["attention_mask"]

        # Get answer details if available
        answer_start_char = -1
        answer_end_char = -1
        if split != "test":
            answer_text = str(row["answer_text"])
            answer_start_char = int(row["answer_start"])
            answer_end_char = answer_start_char + len(answer_text)

        # Iterate over generated windows
        for i, offsets in enumerate(offset_mappings):
            sequence_ids = tokenized.sequence_ids(i)
            input_ids = input_ids_list[i]
            attention_mask = attention_mask_list[i]

            # Create feature dict
            feature = {
                "example_id": example_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,  # List of tuples/lists
                "context": context,
            }

            if split != "test":
                # Initialize labels with -100 (Ignore)
                labels = [-100] * config.max_length

                # Labeling Logic (Soft Overlap)
                for token_idx, (start, end) in enumerate(offsets):
                    # Skip special tokens and question tokens
                    # sequence_ids: None (special), 0 (question), 1 (context)
                    if sequence_ids[token_idx] != 1:
                        continue

                    # Skip special tokens with (0,0) offset mapping if any remain
                    if start == end:
                        continue

                    # Check overlap with answer
                    # Overlap exists if the token span intersects the answer span
                    # Logic: start < answer_end AND end > answer_start
                    if start < answer_end_char and end > answer_start_char:
                        # Determine B-ANS (1) vs I-ANS (2)
                        # If this token contains the start of the answer, mark as B
                        if start <= answer_start_char < end:
                            labels[token_idx] = 1  # B-ANS
                        else:
                            labels[token_idx] = 2  # I-ANS
                    else:
                        # Context token but not answer
                        labels[token_idx] = 0  # O

                feature["labels"] = labels

            features.append(feature)

    # 5. Create DataFrame and Save
    df_features = pd.DataFrame(features)

    os.makedirs(config.cache_dir, exist_ok=True)
    # Save to Parquet (Pandas handles list columns automatically)
    df_features.to_parquet(cache_path, index=False)

    print(f"Saved {len(df_features)} features to {cache_path}")
    return df_features


def prepare_train_features(
    config: Config, tokenizer: PreTrainedTokenizerBase, load_cached_data: bool = True
) -> pd.DataFrame:
    """Wrapper for preparing training features."""
    return prepare_features(
        config, tokenizer, split="train", load_cached_data=load_cached_data
    )
