import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from library.config import Config
from library.utils import split_sentences


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering Token Classification.
    """

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert encoding lists to tensors
        # Exclude sample_idx from tensor conversion if it's not needed by model,
        # but we need it for aggregation. We'll return it as a tensor.
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
            if key != "offset_mapping"
        }
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def process_data(df, tokenizer, mode="train"):
    """
    Converts a pandas DataFrame into a QADataset using Sliding Window.
    Cite solution_lesson_node_00002: Avoid unsupervised hard filtering... employ a Sliding Window approach.

    Args:
        df (pd.DataFrame): The data.
        tokenizer: Hugging Face tokenizer.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        QADataset: The processed dataset.
    """
    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    # Tokenize with sliding window
    encodings = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=Config.MAX_LEN,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # overflow_to_sample_mapping maps each feature to the index of the example in df
    sample_mapping = encodings.pop("overflow_to_sample_mapping")
    # Store sample_idx in encodings to track which window belongs to which example
    encodings["sample_idx"] = sample_mapping

    labels = None

    if mode in ["train", "val"]:
        labels = []
        offset_mappings = encodings["offset_mapping"]

        for i, offsets in enumerate(offset_mappings):
            sample_idx = sample_mapping[i]

            # Get ground truth for this example
            answer_text = df.iloc[sample_idx]["answer_text"]
            start_char = df.iloc[sample_idx]["answer_start"]
            end_char = start_char + len(answer_text)

            sequence_ids = encodings.sequence_ids(i)

            # Initialize labels with O
            label_ids = [Config.LABELS_TO_IDS["O"]] * len(sequence_ids)

            # Find the start and end of the context in the current window
            # We need to check if the answer is fully contained in this window

            # Get the span of context tokens
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(sequence_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in this window)
            # offsets[token_start_index] is the (start, end) char of the first context token
            # offsets[token_end_index] is the (start, end) char of the last context token

            window_start_char = offsets[token_start_index][0]
            window_end_char = offsets[token_end_index][1]

            # If the answer is not fully inside the context of this window, label as O
            # (We could handle partials, but strict containment is cleaner for training)
            if not (window_start_char <= start_char and end_char <= window_end_char):
                labels.append(label_ids)
                continue

            # If contained, find the start and end token indices
            # We iterate through tokens to find the match

            found_start = False

            for idx in range(token_start_index, token_end_index + 1):
                token_start, token_end = offsets[idx]

                # Check overlap
                # We want tokens that are part of the answer
                if token_start < end_char and token_end > start_char:
                    if not found_start:
                        label_ids[idx] = Config.LABELS_TO_IDS["B-ANS"]
                        found_start = True
                    else:
                        label_ids[idx] = Config.LABELS_TO_IDS["I-ANS"]

            labels.append(label_ids)

    return QADataset(encodings, labels)


def prepare_data(load_cached_data=True):
    """
    Loads raw data, processes it, and returns PyTorch Datasets.
    Uses caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache = os.path.join(Config.WORKING_DIR, "train_data.pt")
    val_cache = os.path.join(Config.WORKING_DIR, "val_data.pt")
    test_cache = os.path.join(Config.WORKING_DIR, "test_data.pt")

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading cached datasets from {Config.WORKING_DIR}...")
            train_dataset = torch.load(train_cache)
            val_dataset = torch.load(val_cache)
            test_dataset = torch.load(test_cache)
            return train_dataset, val_dataset, test_dataset

    # 2. Process from Scratch
    print("Processing datasets from scratch...")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Apply Debugging Limits
    if Config.DEBUG:
        print(f"Debug mode: Limiting data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Process Splits
    train_dataset = process_data(df_train, tokenizer, mode="train")
    val_dataset = process_data(df_val, tokenizer, mode="val")
    test_dataset = process_data(df_test, tokenizer, mode="test")

    # 3. Save to Cache
    print(f"Saving datasets to {Config.WORKING_DIR}...")
    torch.save(train_dataset, train_cache)
    torch.save(val_dataset, val_cache)
    torch.save(test_dataset, test_cache)

    return train_dataset, val_dataset, test_dataset
