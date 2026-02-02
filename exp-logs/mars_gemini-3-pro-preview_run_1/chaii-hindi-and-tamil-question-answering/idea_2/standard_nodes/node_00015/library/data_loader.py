import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed

# Suppress tokenizer warnings for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QADataset(Dataset):
    """
    PyTorch Dataset for Extractive Question Answering.
    Handles Training (Inputs + Labels), Validation (Inputs + Labels + Metadata),
    and Test (Inputs + Metadata) modes.
    """

    def __init__(self, data_df, mode="train"):
        """
        Args:
            data_df (pd.DataFrame): The processed dataframe containing features.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data_df
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Basic Inputs
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        # Labels for Training and Validation (if available)
        if self.mode in ["train", "val"]:
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)

        # Metadata for Inference (Validation and Test)
        if self.mode in ["val", "test"]:
            item["example_id"] = row["example_id"]
            # offset_mapping is stored as a list of lists in parquet
            offset_mapping = row["offset_mapping"]
            if isinstance(offset_mapping, np.ndarray):
                offset_mapping = offset_mapping.tolist()
            # Ensure it is a list of lists, not list of numpy arrays (Cite solution_lesson_node_00010)
            if (
                isinstance(offset_mapping, list)
                and len(offset_mapping) > 0
                and isinstance(offset_mapping[0], np.ndarray)
            ):
                offset_mapping = [x.tolist() for x in offset_mapping]
            item["offset_mapping"] = torch.tensor(offset_mapping, dtype=torch.long)

        return item


def prepare_features(df, tokenizer, mode="train"):
    """
    Processes the dataframe into sliding window features compatible with the model.

    Args:
        df (pd.DataFrame): Raw data containing context, question, etc.
        tokenizer: The HuggingFace tokenizer.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: Processed features.
    """
    # Clean questions (remove leading/trailing whitespace)
    df["question"] = df["question"].apply(lambda x: str(x).strip())

    # Tokenization with sliding window
    # We pad to max_length to ensure consistent tensor shapes
    tokenized_examples = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        truncation="only_second",  # Truncate context, not question
        max_length=Config.max_length,
        stride=Config.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    # Iterate over each generated window
    for i, offsets in enumerate(
        tqdm(offset_mapping, desc=f"Processing {mode} features", disable=True)
    ):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # Map window back to the original example index
        sample_index = sample_mapping[i]
        example_row = df.iloc[sample_index]

        # Base feature dictionary
        feature = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add metadata for inference
        if mode in ["val", "test"]:
            feature["example_id"] = example_row["id"]
            # Convert tuples to lists for Parquet compatibility
            feature["offset_mapping"] = [list(o) for o in offsets]

        # Add labels for training/validation
        if mode in ["train", "val"]:
            # If no answer is provided (should not happen in train/val based on dataset spec), point to CLS
            if pd.isna(example_row["answer_start"]):
                feature["start_positions"] = 0
                feature["end_positions"] = 0
            else:
                start_char = example_row["answer_start"]
                end_char = start_char + len(example_row["answer_text"])

                # Sequence IDs help distinguish Question (0) from Context (1)
                # XLM-R: <s> Question </s> </s> Context </s>
                # sequence_ids: None, 0, ..., None, None, 1, ..., None
                sequence_ids = tokenized_examples.sequence_ids(i)

                # Find the start and end of the context in the current window
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx

                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # Check if the answer is fully contained in this window
                # offsets[context_start][0] is the start char of the first context token
                # offsets[context_end][1] is the end char of the last context token
                if not (
                    offsets[context_start][0] <= start_char
                    and offsets[context_end][1] >= end_char
                ):
                    # Answer not in this window -> Label as CLS (0)
                    feature["start_positions"] = 0
                    feature["end_positions"] = 0
                else:
                    # Map character positions to token positions
                    idx = context_start
                    while idx <= context_end and offsets[idx][0] <= start_char:
                        idx += 1
                    feature["start_positions"] = idx - 1

                    idx = context_start
                    while idx <= context_end and offsets[idx][1] < end_char:
                        idx += 1
                    feature["end_positions"] = idx

        features.append(feature)

    processed_df = pd.DataFrame(features)

    # Negative Sampling (Only for Training)
    if mode == "train":
        positives = processed_df[
            (processed_df["start_positions"] != 0)
            | (processed_df["end_positions"] != 0)
        ]
        negatives = processed_df[
            (processed_df["start_positions"] == 0)
            & (processed_df["end_positions"] == 0)
        ]

        # Calculate number of negatives to keep
        n_pos = len(positives)
        n_neg = int(n_pos * Config.negative_sampling_ratio)

        # Downsample negatives if we have more than required
        if len(negatives) > n_neg:
            negatives = negatives.sample(n=n_neg, random_state=Config.seed)

        # Combine and shuffle
        processed_df = (
            pd.concat([positives, negatives])
            .sample(frac=1, random_state=Config.seed)
            .reset_index(drop=True)
        )

    return processed_df


def get_processed_data(tokenizer, load_cached_data=True):
    """
    Loads raw data, processes it into features (or loads from cache),
    and returns DataFrames for train, val, and test.

    Args:
        tokenizer: The tokenizer to use for processing.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed(Config.seed)

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Define cache paths
    train_cache = Config.train_cache_path
    val_cache = Config.val_cache_path
    test_cache = Config.test_cache_path

    # --- Train Data ---
    if load_cached_data and os.path.exists(train_cache):
        print(f"Loading cached training data from {train_cache}")
        train_df = pd.read_parquet(train_cache)
    else:
        print("Processing training data...")
        if Config.debug:
            raw_train = pd.read_csv(Config.train_path).head(Config.debug_subset_size)
        else:
            raw_train = pd.read_csv(Config.train_path)

        train_df = prepare_features(raw_train, tokenizer, mode="train")
        train_df.to_parquet(train_cache)

    # --- Validation Data ---
    if load_cached_data and os.path.exists(val_cache):
        print(f"Loading cached validation data from {val_cache}")
        val_df = pd.read_parquet(val_cache)
    else:
        print("Processing validation data...")
        if Config.debug:
            raw_val = pd.read_csv(Config.val_path).head(Config.debug_subset_size)
        else:
            raw_val = pd.read_csv(Config.val_path)

        val_df = prepare_features(raw_val, tokenizer, mode="val")
        val_df.to_parquet(val_cache)

    # --- Test Data ---
    if load_cached_data and os.path.exists(test_cache):
        print(f"Loading cached test data from {test_cache}")
        test_df = pd.read_parquet(test_cache)
    else:
        print("Processing test data...")
        if Config.debug:
            raw_test = pd.read_csv(Config.test_path).head(Config.debug_subset_size)
        else:
            raw_test = pd.read_csv(Config.test_path)

        test_df = prepare_features(raw_test, tokenizer, mode="test")
        test_df.to_parquet(test_cache)

    return train_df, val_df, test_df
