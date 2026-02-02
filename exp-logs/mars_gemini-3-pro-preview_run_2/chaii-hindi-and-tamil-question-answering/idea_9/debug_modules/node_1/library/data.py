import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEEDS[0])


def get_tokenizer():
    """Loads the tokenizer defined in Config."""
    return AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    """

    def __init__(self, features_df, mode="train"):
        self.features = features_df
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        row = self.features.iloc[idx]

        # Common inputs
        item = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }

        # Training targets
        if self.mode == "train":
            item["start_positions"] = torch.tensor(
                row["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(row["end_positions"], dtype=torch.long)
            item["answerable"] = torch.tensor(row["answerable"], dtype=torch.float)

        # For inference, we might need example_id to map back,
        # but usually that's handled by the loop using the raw features list.
        # We can return the index to map back if needed.
        item["index"] = torch.tensor(idx, dtype=torch.long)

        return item


def prepare_train_features(examples, tokenizer):
    """
    Tokenizes training data with sliding windows and maps character answers to token positions.
    """
    # Clean whitespace
    examples["question"] = examples["question"].str.strip()

    # Tokenize
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    # Convert dataframe to list of dicts for faster access
    example_records = examples.to_dict("records")

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        # Get original example
        sample_index = sample_mapping[i]
        example = example_records[sample_index]

        # Sequence IDs: None (special), 0 (question), 1 (context)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find context bounds
        # We need to find the start and end of the context in the input_ids
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
            if token_start_index >= len(sequence_ids):
                break

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
            if token_end_index < 0:
                break

        # Detect answer
        answer_text = example["answer_text"]
        start_char = example["answer_start"]
        end_char = start_char + len(answer_text)

        # Check if answer is within the context window
        # Offsets are (start_char, end_char) for each token
        # If the context window doesn't cover the answer span, label as unanswerable (0,0)

        # Bounds of the current window's context
        context_start_char = offsets[token_start_index][0]
        context_end_char = offsets[token_end_index][1]

        if not (context_start_char <= start_char and context_end_char >= end_char):
            start_position = 0
            end_position = 0
            answerable = 0.0
        else:
            # Answer is within the window
            # Move token_start_index to the start of the answer
            idx = token_start_index
            while idx <= token_end_index and offsets[idx][1] <= start_char:
                idx += 1
            start_position = idx

            # Move token_end_index to the end of the answer
            idx = token_end_index
            while idx >= token_start_index and offsets[idx][0] >= end_char:
                idx -= 1
            end_position = idx

            answerable = 1.0

        # Create offset mapping with None for non-context tokens (consistent with test features)
        final_offsets = []
        for k, o in enumerate(offsets):
            if sequence_ids[k] == 1:
                final_offsets.append(list(o))
            else:
                final_offsets.append(None)

        features.append(
            {
                "example_id": example["id"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
                "answerable": answerable,
                "offset_mapping": final_offsets,
            }
        )

    return pd.DataFrame(features)


def prepare_test_features(examples, tokenizer):
    """
    Tokenizes test data with sliding windows. Preserves offset mapping for post-processing.
    """
    examples["question"] = examples["question"].str.strip()

    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []
    example_records = examples.to_dict("records")

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]

        sample_index = sample_mapping[i]
        example = example_records[sample_index]

        # We need to set offset_mapping to None for non-context tokens
        # to avoid predicting answers in the question or special tokens
        sequence_ids = tokenized_examples.sequence_ids(i)

        final_offsets = []
        for k, o in enumerate(offsets):
            if sequence_ids[k] == 1:
                final_offsets.append(
                    list(o)
                )  # Convert tuple to list for Parquet compatibility
            else:
                final_offsets.append(None)  # None indicates non-context

        features.append(
            {
                "example_id": example["id"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": final_offsets,
            }
        )

    return pd.DataFrame(features)


def process_and_cache_features(split, examples, tokenizer, load_cached_data=True):
    """
    Handles caching logic: Load from parquet if exists, else process and save.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{split}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            # Parquet might load lists as numpy arrays, ensure they are lists if needed
            # But usually for creating tensors, numpy arrays are fine.
            # For offset_mapping, we need to ensure None is handled correctly (Parquet handles nulls)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {split} features...")
    if split in ["train", "val"]:
        # Validation set in this context also has answers, so we treat it like train
        # unless we want to evaluate on it using the model's inference mode.
        # However, for the training loop validation, we typically want the targets (start/end).
        # If we want to run full inference on validation, we might need a separate 'val_inference' split logic.
        # For now, we assume 'val' is used for loss calculation or we use prepare_train_features.
        # If the input csv has 'answer_text', we use prepare_train_features.
        if "answer_text" in examples.columns:
            df = prepare_train_features(examples, tokenizer)
        else:
            df = prepare_test_features(examples, tokenizer)
    else:
        df = prepare_test_features(examples, tokenizer)

    print(f"Saving {len(df)} features to {cache_path}...")
    # Ensure lists are compatible with parquet
    # Pandas to_parquet with pyarrow engine handles lists of primitives well.
    # offset_mapping contains None, which maps to null in parquet.
    df.to_parquet(cache_path, engine="pyarrow", index=False)

    return df


def get_processed_data(split="train", load_cached_data=True, debug=False):
    """
    Main entry point to get the Dataset and raw features.

    Args:
        split: 'train', 'val', or 'test'
        load_cached_data: Whether to use cached parquet files.
        debug: If True, subsamples the data for quick testing.

    Returns:
        dataset: QADataset object
        features_list: List of dictionaries (raw features for post-processing)
    """
    # 1. Identify Input File
    if split == "train":
        file_path = Config.TRAIN_PATH
    elif split == "val":
        file_path = Config.VAL_PATH
    elif split == "test":
        file_path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # 2. Load Raw Data
    df_raw = pd.read_csv(file_path)

    if debug:
        print(f"DEBUG MODE: Subsampling {split} data...")
        df_raw = df_raw.head(Config.DEBUG_SAMPLE_SIZE)
        # Modify cache path for debug to avoid overwriting full cache
        original_working_dir = Config.WORKING_DIR
        Config.WORKING_DIR = os.path.join(Config.WORKING_DIR, "debug")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 3. Process Features
    tokenizer = get_tokenizer()

    # We pass the split name. If split is 'val' but has answers, process_and_cache handles it.
    df_features = process_and_cache_features(split, df_raw, tokenizer, load_cached_data)

    # 4. Create Dataset
    # Determine mode for Dataset
    # If we have targets, mode is train (returns labels). Else mode is test.
    # Note: Validation set has targets, so we can return labels for evaluation loss.
    has_targets = "start_positions" in df_features.columns
    mode = "train" if has_targets else "test"

    dataset = QADataset(df_features, mode=mode)

    # 5. Prepare raw features list for post-processing
    # Converting to list of dicts is efficient for the sequential access in post-processing
    features_list = df_features.to_dict("records")

    if debug:
        # Restore working dir
        Config.WORKING_DIR = original_working_dir

    return dataset, features_list
