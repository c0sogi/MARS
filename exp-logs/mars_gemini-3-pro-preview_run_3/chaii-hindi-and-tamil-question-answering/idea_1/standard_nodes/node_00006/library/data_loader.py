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
    PyTorch Dataset for Question Answering Token Classification with Sliding Window.
    """

    def __init__(self, encodings, labels=None, sample_indices=None):
        self.encodings = encodings
        self.labels = labels
        self.sample_indices = sample_indices

    def __getitem__(self, idx):
        # Convert encoding lists to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.sample_indices is not None:
            item["sample_idx"] = torch.tensor(
                self.sample_indices[idx], dtype=torch.long
            )
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def process_data(df, tokenizer, mode="train"):
    """
    Converts a pandas DataFrame into a QADataset using sliding window.
    Cite solution_lesson_node_00002: Replaces hard retrieval with sliding window.
    """
    questions = df["question"].astype(str).tolist()
    contexts = df["context"].astype(str).tolist()

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

    # Map chunks back to original examples
    sample_mapping = encodings.pop("overflow_to_sample_mapping")
    offset_mapping = encodings.pop("offset_mapping")

    labels = None

    if mode in ["train", "val"]:
        labels = []
        answers = df["answer_text"].astype(str).tolist()
        answer_starts = df["answer_start"].tolist()

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer_text = answers[sample_idx]
            start_char = answer_starts[sample_idx]
            end_char = start_char + len(answer_text)

            sequence_ids = encodings.sequence_ids(i)

            # Initialize labels with O
            label_ids = [Config.LABELS_TO_IDS["O"]] * len(sequence_ids)

            # Find the context start and end indices in the token list
            # sequence_ids: None (special), 0 (question), 1 (context)
            context_token_indices = [
                idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if not context_token_indices:
                labels.append(label_ids)
                continue

            # Check if answer is fully contained in this chunk
            # offsets[idx] is (start, end) in original context string
            chunk_start_char = offsets[context_token_indices[0]][0]
            chunk_end_char = offsets[context_token_indices[-1]][1]

            # If answer is within this chunk
            if start_char >= chunk_start_char and end_char <= chunk_end_char:
                found_start = False
                for idx in context_token_indices:
                    token_start, token_end = offsets[idx]

                    # Check overlap with answer span
                    if token_start < end_char and token_end > start_char:
                        if not found_start:
                            label_ids[idx] = Config.LABELS_TO_IDS["B-ANS"]
                            found_start = True
                        else:
                            label_ids[idx] = Config.LABELS_TO_IDS["I-ANS"]

            labels.append(label_ids)

    return QADataset(encodings, labels, sample_indices=sample_mapping)


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
