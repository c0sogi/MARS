import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything


class QADataset(Dataset):
    """
    PyTorch Dataset for the Hindi/Tamil QA Task.
    Handles serving of tokenized features for both training and inference.
    """

    def __init__(self, data: pd.DataFrame, mode: str = "train"):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the processed features.
            mode (str): 'train' or 'test'. Determines which fields are returned.
        """
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Common inputs
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long)

        if self.mode == "train":
            # Training targets
            start_position = torch.tensor(row["start_position"], dtype=torch.long)
            end_position = torch.tensor(row["end_position"], dtype=torch.long)
            relevance = torch.tensor(row["relevance"], dtype=torch.float)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "relevance_labels": relevance,
            }
        else:
            # Inference metadata
            # offset_mapping is stored as a list of lists in parquet, convert to tensor
            offset_mapping = torch.tensor(row["offset_mapping"], dtype=torch.long)
            example_id = row["example_id"]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offset_mapping,
                "example_id": example_id,
            }


def process_train_data(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, config: Config
) -> pd.DataFrame:
    """
    Processes raw training data into sliding window features with negative sampling.
    """
    # Ensure inputs are strings
    df["question"] = df["question"].astype(str)
    df["context"] = df["context"].astype(str)
    df["answer_text"] = df["answer_text"].astype(str)

    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=config.max_length,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if config.pad_to_max_length else False,
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
        row = df.iloc[sample_index]

        answer_start_char = row["answer_start"]
        answer_text = row["answer_text"]
        answer_end_char = answer_start_char + len(answer_text)

        # Find context token boundaries
        # sequence_ids: None (special), 0 (question), None, 1 (context), None
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if answer is inside this window
        # offsets[index] is (start_char, end_char)
        context_start_char = offsets[token_start_index][0]
        context_end_char = offsets[token_end_index][1]

        is_answer_contained = (offsets[token_start_index][0] <= answer_start_char) and (
            offsets[token_end_index][1] >= answer_end_char
        )

        if not is_answer_contained:
            # Negative sample (or partial answer treated as negative)
            start_position = 0  # CLS token
            end_position = 0  # CLS token
            relevance = 0
        else:
            # Positive sample: Find token indices
            # Move start token index forward to answer start
            current_idx = token_start_index
            while (
                current_idx <= token_end_index
                and offsets[current_idx][0] <= answer_start_char
            ):
                current_idx += 1
            start_position = current_idx - 1

            # Move end token index backward to answer end
            current_idx = token_end_index
            while (
                current_idx >= token_start_index
                and offsets[current_idx][1] >= answer_end_char
            ):
                current_idx -= 1
            end_position = current_idx + 1

            relevance = 1

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_position": start_position,
                "end_position": end_position,
                "relevance": relevance,
                "example_id": row["id"],
            }
        )

    features_df = pd.DataFrame(features)

    # Negative Sampling
    # Separate positives and negatives
    pos_df = features_df[features_df["relevance"] == 1]
    neg_df = features_df[features_df["relevance"] == 0]

    n_pos = len(pos_df)
    n_neg_target = int(n_pos * config.negative_positive_ratio)

    # Downsample negatives if we have more than needed
    if len(neg_df) > n_neg_target:
        # Use a fixed random state for reproducibility of the dataset creation
        rng = np.random.RandomState(config.seed)
        neg_df = neg_df.sample(n=n_neg_target, random_state=rng)

    # Combine and shuffle
    final_df = pd.concat([pos_df, neg_df])
    final_df = final_df.sample(
        frac=1, random_state=np.random.RandomState(config.seed)
    ).reset_index(drop=True)

    return final_df


def process_test_data(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, config: Config
) -> pd.DataFrame:
    """
    Processes raw test data into sliding window features. Keeps all windows.
    """
    df["question"] = df["question"].astype(str)
    df["context"] = df["context"].astype(str)

    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=config.max_length,
        stride=config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if config.pad_to_max_length else False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        example_id = df.iloc[sample_index]["id"]

        # We set non-context offsets to None or (0,0) to avoid confusion during inference post-processing
        # though standard offset_mapping usually handles this.
        # We just store the raw offset mapping for the context tokens.

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,  # List of (int, int) tuples
                "example_id": example_id,
            }
        )

    return pd.DataFrame(features)


def get_train_dataset(
    config: Config, tokenizer: PreTrainedTokenizerBase, load_cached_data: bool = True
) -> QADataset:
    """
    Loads, processes, and caches the training dataset.
    Merges Train and Validation data as per the Full-Data strategy.
    """
    cache_path = os.path.join(config.cache_dir, "train_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training features from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Generating training features...")
        # Load metadata CSVs
        train_df = pd.read_csv(config.train_path)
        val_df = pd.read_csv(config.val_path)

        # Merge for Full-Data training
        full_df = pd.concat([train_df, val_df], ignore_index=True)

        if config.debug:
            full_df = full_df.head(config.debug_sample_size)

        # Process
        df = process_train_data(full_df, tokenizer, config)

        # Cache
        print(f"Saving training features to {cache_path}")
        # Parquet handles lists in columns well
        df.to_parquet(cache_path, index=False)

    return QADataset(df, mode="train")


def get_test_dataset(
    config: Config, tokenizer: PreTrainedTokenizerBase, load_cached_data: bool = True
) -> QADataset:
    """
    Loads, processes, and caches the test dataset.
    """
    cache_path = os.path.join(config.cache_dir, "test_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test features from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Generating test features...")
        test_df = pd.read_csv(config.test_path)

        if config.debug:
            test_df = test_df.head(config.debug_sample_size)

        df = process_test_data(test_df, tokenizer, config)

        print(f"Saving test features to {cache_path}")
        df.to_parquet(cache_path, index=False)

    return QADataset(df, mode="test")
