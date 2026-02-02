import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.utils import seed_everything


class QADataset(Dataset):
    """
    PyTorch Dataset for serving processed Question Answering features.
    Handles both training (with targets) and testing (without targets) modes.
    """

    def __init__(self, data, is_test=False):
        """
        Args:
            data (pd.DataFrame): DataFrame containing processed features.
            is_test (bool): Whether the dataset is for inference (no labels).
        """
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Convert list features to tensors
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        if not self.is_test:
            # Training mode: return start and end positions
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)
        else:
            # Inference mode: return metadata for reconstruction
            item["example_id"] = row["example_id"]

            # Normalize offset_mapping (handle both list and np.ndarray)
            offset_mapping = row["offset_mapping"]
            if isinstance(offset_mapping, np.ndarray):
                offset_mapping = offset_mapping.tolist()
            item["offset_mapping"] = torch.tensor(offset_mapping, dtype=torch.long)

            # sequence_ids helps identify context tokens during post-processing
            sequence_ids = row["sequence_ids"]
            if isinstance(sequence_ids, np.ndarray):
                sequence_ids = sequence_ids.tolist()
            item["sequence_ids"] = torch.tensor(sequence_ids, dtype=torch.long)

        return item


def get_folds(df, config):
    """
    Performs GroupKFold splitting on the dataframe.
    Groups by 'context' to ensure no context leakage between folds.

    Args:
        df (pd.DataFrame): Raw training dataframe.
        config (Config): Configuration object containing n_folds.

    Returns:
        pd.DataFrame: Dataframe with a new 'fold' column.
    """
    gkf = GroupKFold(n_splits=config.n_folds)
    df["fold"] = -1
    for fold_num, (train_idx, val_idx) in enumerate(
        gkf.split(df, groups=df["context"])
    ):
        df.loc[val_idx, "fold"] = fold_num
    return df


def prepare_train_features(examples, tokenizer, config):
    """
    Tokenizes training examples with sliding windows and maps answers to token positions.

    Args:
        examples (pd.DataFrame): Raw training examples.
        tokenizer: HuggingFace tokenizer.
        config (Config): Configuration object.

    Returns:
        pd.DataFrame: Processed features including input_ids, masks, and targets.
    """
    # Tokenize with sliding window (stride)
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=config.max_length,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    # Convert columns to lists for faster access
    ids = examples["id"].tolist()
    answer_texts = examples["answer_text"].tolist()
    answer_starts = examples["answer_start"].tolist()
    folds = examples["fold"].tolist()

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # Map back to the original example index
        sample_index = sample_mapping[i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Get answer character positions
        start_char = answer_starts[sample_index]
        end_char = start_char + len(answer_texts[sample_index])

        # Determine the start and end token indices of the context
        # sequence_ids: None (special), 0 (question), 1 (context)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Check if the answer is fully contained in this window
        # offsets[idx] is (start_char, end_char) of the token
        if not (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            # Answer is not inside the window, label as 0 (CLS)
            start_position = 0
            end_position = 0
        else:
            # Move token_start_index to the start of the answer
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            start_position = token_start_index - 1

            # Move token_end_index to the end of the answer
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_position = token_end_index + 1

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "fold": folds[sample_index],
                "example_id": ids[sample_index],
            }
        )

    return pd.DataFrame(features)


def prepare_test_features(examples, tokenizer, config):
    """
    Tokenizes test examples with sliding windows for inference.

    Args:
        examples (pd.DataFrame): Raw test examples.
        tokenizer: HuggingFace tokenizer.
        config (Config): Configuration object.

    Returns:
        pd.DataFrame: Processed features with offset mappings for reconstruction.
    """
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=config.max_length,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    features = []
    ids = examples["id"].tolist()

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sample_index = sample_mapping[i]

        sequence_ids = tokenized_examples.sequence_ids(i)
        # Replace None with -1 to make it integer-compatible for storage/tensor conversion
        seq_ids_clean = [s if s is not None else -1 for s in sequence_ids]

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,
                "example_id": ids[sample_index],
                "sequence_ids": seq_ids_clean,
            }
        )

    return pd.DataFrame(features)


def load_and_process_data(config, tokenizer, load_cached_data=True):
    """
    Orchestrates data loading, fold assignment, preprocessing, and caching.

    Args:
        config (Config): Configuration object.
        tokenizer: HuggingFace tokenizer.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_features, val_features, test_features) as DataFrames.
    """
    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # --- 1. Train Data ---
    train_features = None
    if load_cached_data and os.path.exists(config.train_cache_path):
        train_features = pd.read_parquet(config.train_cache_path)
        if "fold" not in train_features.columns:
            train_features = None

    if train_features is None:
        # Load raw metadata
        train_df = pd.read_csv(config.train_data_path)

        # Assign folds
        train_df = get_folds(train_df, config)

        # Process into windows
        train_features = prepare_train_features(train_df, tokenizer, config)

        # Cache
        train_features.to_parquet(config.train_cache_path, index=False)

    # --- 2. Validation Data ---
    # Note: We typically use folds from train_features for CV, but we process the
    # explicit validation set from metadata if it exists for global validation.
    if load_cached_data and os.path.exists(config.val_cache_path):
        val_features = pd.read_parquet(config.val_cache_path)
    else:
        if os.path.exists(config.val_data_path):
            val_df = pd.read_csv(config.val_data_path)
            # Assign dummy fold
            val_df["fold"] = -1
            val_features = prepare_train_features(val_df, tokenizer, config)
            val_features.to_parquet(config.val_cache_path, index=False)
        else:
            val_features = pd.DataFrame()

    # --- 3. Test Data ---
    if load_cached_data and os.path.exists(config.test_cache_path):
        test_features = pd.read_parquet(config.test_cache_path)
    else:
        test_df = pd.read_csv(config.test_data_path)
        test_features = prepare_test_features(test_df, tokenizer, config)
        test_features.to_parquet(config.test_cache_path, index=False)

    return train_features, val_features, test_features
