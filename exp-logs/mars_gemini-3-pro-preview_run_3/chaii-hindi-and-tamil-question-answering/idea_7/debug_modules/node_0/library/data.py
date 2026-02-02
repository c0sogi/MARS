import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


def get_tokenizer():
    """Returns the tokenizer defined in Config."""
    return AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)


def prepare_train_features(examples, tokenizer):
    """
    Tokenizes examples with sliding window and generates labels using Soft Overlap.

    Args:
        examples (pd.DataFrame): DataFrame containing 'question', 'context',
                                 'answer_text', 'answer_start'.
        tokenizer: The HuggingFace tokenizer.

    Returns:
        list[dict]: A list of feature dictionaries.
    """
    # Convert DataFrame to dict of lists for tokenizer
    examples_dict = examples.to_dict(orient="list")

    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        examples_dict["question"],
        examples_dict["context"],
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offset_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Map back to original example
        sample_index = sample_mapping[i]
        example_id = examples_dict["id"][sample_index]
        context_text = examples_dict["context"][sample_index]
        answer_text = examples_dict["answer_text"][sample_index]
        answer_start = examples_dict["answer_start"][sample_index]

        # Determine answer end
        answer_end = answer_start + len(answer_text)

        # Initialize labels with O (0)
        labels = [0] * len(input_ids)

        # Soft Overlap Labeling
        # We only label tokens that are part of the context (sequence_id == 1)
        for idx, (seq_id, (start_char, end_char)) in enumerate(
            zip(sequence_ids, offsets)
        ):
            if seq_id != 1:
                continue

            # Skip special tokens (offset is usually 0,0)
            if start_char == 0 and end_char == 0:
                continue

            # Check overlap
            # Overlap exists if max(start_char, answer_start) < min(end_char, answer_end)
            overlap_start = max(start_char, answer_start)
            overlap_end = min(end_char, answer_end)

            if overlap_start < overlap_end:
                # Overlap detected
                # Determine if B-ANS (1) or I-ANS (2)
                # If the token covers the start of the answer, mark as B-ANS
                # Note: We check if the answer starts within this token's span
                if start_char <= answer_start < end_char:
                    labels[idx] = 1  # B-ANS
                else:
                    labels[idx] = 2  # I-ANS

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "example_id": example_id,
                "offset_mapping": offsets,
                "context": context_text,
                "sequence_ids": sequence_ids,
            }
        )

    return features


def prepare_test_features(examples, tokenizer):
    """
    Tokenizes examples with sliding window for inference (no labels).
    """
    examples_dict = examples.to_dict(orient="list")

    tokenized_examples = tokenizer(
        examples_dict["question"],
        examples_dict["context"],
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offset_mapping=True,
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
        example_id = examples_dict["id"][sample_index]
        context_text = examples_dict["context"][sample_index]

        # Dummy labels
        labels = [0] * len(input_ids)

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,  # Not used for inference but kept for consistency
                "example_id": example_id,
                "offset_mapping": offsets,
                "context": context_text,
                "sequence_ids": sequence_ids,
            }
        )

    return features


class QADataset(Dataset):
    def __init__(self, mode="train", load_cached_data=Config.LOAD_CACHED_DATA):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from parquet cache.
        """
        self.mode = mode
        self.tokenizer = get_tokenizer()

        # Determine paths
        if mode == "train":
            csv_path = Config.TRAIN_CSV
            cache_name = "train_features.parquet"
        elif mode == "val":
            csv_path = Config.VAL_CSV
            cache_name = "val_features.parquet"
        else:
            csv_path = Config.TEST_CSV
            cache_name = "test_features.parquet"

        cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        # Logic: Load Cache OR Process & Cache
        if load_cached_data and os.path.exists(cache_path):
            # print(f"Loading {mode} features from {cache_path}")
            self.df_features = pd.read_parquet(cache_path, engine="pyarrow")
        else:
            # print(f"Processing {mode} data from {csv_path}...")
            df_raw = pd.read_csv(csv_path)

            # Fill NaNs to avoid errors
            df_raw["question"] = df_raw["question"].fillna("").astype(str)
            df_raw["context"] = df_raw["context"].fillna("").astype(str)
            if "answer_text" in df_raw.columns:
                df_raw["answer_text"] = df_raw["answer_text"].fillna("").astype(str)
                df_raw["answer_start"] = df_raw["answer_start"].fillna(0).astype(int)

            if mode in ["train", "val"]:
                features_list = prepare_train_features(df_raw, self.tokenizer)
            else:
                features_list = prepare_test_features(df_raw, self.tokenizer)

            # Convert to DataFrame for caching
            self.df_features = pd.DataFrame(features_list)

            # Ensure cache dir exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.df_features.to_parquet(cache_path, engine="pyarrow")

    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, idx):
        # Retrieve row
        row = self.df_features.iloc[idx]

        # Convert list/array columns back to tensors/objects
        # Note: Parquet stores lists as numpy arrays of objects or lists.
        # We need to ensure they are in the right format for the collate_fn/model.

        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
            "example_id": row["example_id"],
            "offset_mapping": (
                row["offset_mapping"].tolist()
                if isinstance(row["offset_mapping"], np.ndarray)
                else row["offset_mapping"]
            ),
            "context": row["context"],
            "sequence_ids": (
                row["sequence_ids"].tolist()
                if isinstance(row["sequence_ids"], np.ndarray)
                else row["sequence_ids"]
            ),
        }


def qa_collate_fn(batch):
    """
    Custom collate function to bundle tensors and metadata.
    """
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    # Collect metadata
    metadata = []
    for item in batch:
        metadata.append(
            {
                "example_id": item["example_id"],
                "offset_mapping": item["offset_mapping"],
                "context": item["context"],
                "sequence_ids": item["sequence_ids"],
            }
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": metadata,
    }


class TAPTDataset(Dataset):
    def __init__(self, load_cached_data=Config.LOAD_CACHED_DATA):
        self.tokenizer = get_tokenizer()
        self.block_size = Config.MAX_LENGTH
        self.stride = Config.DOC_STRIDE

        cache_path = os.path.join(Config.TAPT_CACHE_DIR, "corpus.parquet")

        if load_cached_data and os.path.exists(cache_path):
            # print(f"Loading TAPT data from {cache_path}")
            df = pd.read_parquet(cache_path, engine="pyarrow")
            self.examples = df["input_ids"].tolist()
        else:
            # print("Processing TAPT data...")
            texts = []
            # Load all available text
            for path in [Config.TRAIN_CSV, Config.VAL_CSV, Config.TEST_CSV]:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    texts.extend(df["context"].dropna().astype(str).tolist())

            # Join all text (concatenated context)
            full_text = " ".join(texts)

            # Tokenize with sliding window
            # We treat the entire corpus as one stream
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.block_size,
                stride=self.stride,
                return_overflowing_tokens=True,
                padding="max_length",
                add_special_tokens=True,
            )

            self.examples = tokenized["input_ids"]

            # Cache
            df_cache = pd.DataFrame({"input_ids": self.examples})
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df_cache.to_parquet(cache_path, engine="pyarrow")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
