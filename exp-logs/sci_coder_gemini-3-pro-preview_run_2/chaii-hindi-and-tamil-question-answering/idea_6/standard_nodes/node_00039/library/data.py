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
    PyTorch Dataset for Question Answering.
    Wraps processed features and returns tensors for model training/inference.
    """

    def __init__(self, features_df, mode="train"):
        self.mode = mode
        # Convert columns to lists for efficient indexing
        # Parquet loading might result in numpy arrays, tolist() ensures standard python lists
        self.input_ids = features_df["input_ids"].tolist()
        self.attention_mask = features_df["attention_mask"].tolist()
        self.offset_mapping = features_df["offset_mapping"].tolist()
        self.example_id = features_df["example_id"].tolist()

        if self.mode == "train":
            self.start_positions = features_df["start_positions"].tolist()
            self.end_positions = features_df["end_positions"].tolist()
            # Derive answerable label: 1 if answer is in window (start > 0), 0 otherwise
            self.answerable_label = [1 if s > 0 else 0 for s in self.start_positions]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "offset_mapping": torch.tensor(self.offset_mapping[idx], dtype=torch.long),
            "example_id": self.example_id[idx],
        }

        if self.mode == "train":
            item["start_positions"] = torch.tensor(
                self.start_positions[idx], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(
                self.end_positions[idx], dtype=torch.long
            )
            item["answerable_label"] = torch.tensor(
                self.answerable_label[idx], dtype=torch.float
            )

        return item


def _process_batch(df, tokenizer, config, mode="train"):
    """
    Tokenizes a dataframe of examples using sliding windows and maps answers to token positions.
    Returns a DataFrame where each row is a single window (feature).
    """
    # Tokenize with sliding window
    tokenized_inputs = tokenizer(
        df["question"].tolist(),
        df["context"].tolist(),
        max_length=config.MAX_LENGTH,
        stride=config.DOC_STRIDE,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map from feature index to original sample index
    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    features = []

    for i, offsets in enumerate(offset_mapping):
        feature = {}
        feature["input_ids"] = tokenized_inputs["input_ids"][i]
        feature["attention_mask"] = tokenized_inputs["attention_mask"][i]
        feature["offset_mapping"] = offsets

        sample_idx = sample_mapping[i]
        example = df.iloc[sample_idx]
        feature["example_id"] = example["id"]

        # Propagate fold information if available
        if "fold" in example:
            feature["fold"] = example["fold"]

        if mode == "train":
            # Retrieve ground truth
            answer_text = example["answer_text"]
            start_char = example["answer_start"]
            end_char = start_char + len(answer_text)

            # sequence_ids: None for special tokens, 0 for question, 1 for context
            sequence_ids = tokenized_inputs.sequence_ids(i)

            # Find the start and end of the context in the input_ids
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If no context found (rare edge case), label as CLS (0)
            if context_start > context_end:
                feature["start_positions"] = 0
                feature["end_positions"] = 0
            else:
                # Check if the answer is fully contained in this window
                # offsets[x] = (start_char, end_char)
                window_start_char = offsets[context_start][0]
                window_end_char = offsets[context_end][1]

                if not (
                    window_start_char <= start_char and window_end_char >= end_char
                ):
                    # Answer not fully inside this window -> label as CLS
                    feature["start_positions"] = 0
                    feature["end_positions"] = 0
                else:
                    # Map character position to token position
                    # Find token start
                    idx = context_start
                    while idx <= context_end and offsets[idx][1] <= start_char:
                        idx += 1
                    feature["start_positions"] = idx

                    # Find token end
                    idx = context_end
                    while idx >= context_start and offsets[idx][0] >= end_char:
                        idx -= 1
                    feature["end_positions"] = idx

        features.append(feature)

    return pd.DataFrame(features)


def get_data(load_cached_data=True, debug=False):
    """
    Main function to load, process, and cache data.

    Args:
        load_cached_data (bool): If True, attempts to load processed features from parquet.
        debug (bool): If True, subsamples data for quick testing.

    Returns:
        train_df (pd.DataFrame): Processed training features (with folds).
        test_df (pd.DataFrame): Processed test features.
    """
    seed_everything(Config.SEED)

    # Define cache paths
    cache_dir = Config.OUTPUT_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "cached_train_features.parquet")
    test_cache_path = os.path.join(cache_dir, "cached_test_features.parquet")

    if debug:
        train_cache_path = train_cache_path.replace(
            ".parquet", "_debug_features.parquet"
        )
        test_cache_path = test_cache_path.replace(".parquet", "_debug_features.parquet")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached data from {cache_dir}...")
        try:
            train_features = pd.read_parquet(train_cache_path)
            test_features = pd.read_parquet(test_cache_path)
            return train_features, test_features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print("Processing data from scratch...")

    # Load Metadata
    # We combine metadata/train.csv and metadata/val.csv to perform our own Cross-Validation
    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    df_val_meta = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    df_train = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Debug Subsampling
    if debug:
        print("Debug mode: Subsampling data...")
        # Sample by context to keep groups intact
        contexts = df_train["context"].unique()
        sample_size = (
            20 if Config.DEBUG_SAMPLE_SIZE is None else Config.DEBUG_SAMPLE_SIZE
        )
        sample_contexts = np.random.choice(
            contexts, size=min(sample_size, len(contexts)), replace=False
        )
        df_train = df_train[df_train["context"].isin(sample_contexts)].reset_index(
            drop=True
        )
        df_test = df_test.iloc[:sample_size].reset_index(drop=True)

    # Create Folds
    print(f"Creating {Config.N_FOLDS} folds using GroupKFold...")
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    df_train["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df_train, groups=df_train["context"])
    ):
        df_train.loc[val_idx, "fold"] = fold

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # Process Train
    print("Tokenizing training data (this may take a while)...")
    train_features = _process_batch(df_train, tokenizer, Config, mode="train")

    # Process Test
    print("Tokenizing test data...")
    test_features = _process_batch(df_test, tokenizer, Config, mode="test")

    # Save to Cache
    print(f"Saving processed features to {cache_dir}...")
    train_features.to_parquet(train_cache_path, index=False)
    test_features.to_parquet(test_cache_path, index=False)

    return train_features, test_features


def get_folds(df, fold_idx):
    """
    Splits the processed dataframe into train and validation sets based on the fold index.
    """
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)
    return train_df, val_df
