import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class QuestDataset(Dataset):
    """
    Dataset class for StackExchange Question-Answer pairs.
    Holds pre-tokenized inputs for Question and Answer streams.
    """

    def __init__(
        self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask, labels=None
    ):
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.a_input_ids = a_input_ids
        self.a_attention_mask = a_attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.q_input_ids)

    def __getitem__(self, idx):
        item = {
            "q_input_ids": torch.tensor(self.q_input_ids[idx], dtype=torch.long),
            "q_attention_mask": torch.tensor(
                self.q_attention_mask[idx], dtype=torch.long
            ),
            "a_input_ids": torch.tensor(self.a_input_ids[idx], dtype=torch.long),
            "a_attention_mask": torch.tensor(
                self.a_attention_mask[idx], dtype=torch.long
            ),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def tokenize_stream(texts, tokenizer, max_len):
    """
    Tokenizes a list of text strings into input_ids and attention_masks.
    """
    encoding = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_token_type_ids=False,
    )
    return encoding["input_ids"], encoding["attention_mask"]


def get_dataset(mode, tokenizer, load_cached_data=True):
    """
    Loads data, performs tokenization (with caching), and returns a QuestDataset.

    Args:
        mode (str): 'train', 'val', or 'test'.
        tokenizer: Transformers tokenizer instance.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        QuestDataset: The prepared dataset.
    """
    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    q_ids_path = os.path.join(cache_dir, f"{mode}_q_ids.npy")
    q_mask_path = os.path.join(cache_dir, f"{mode}_q_mask.npy")
    a_ids_path = os.path.join(cache_dir, f"{mode}_a_ids.npy")
    a_mask_path = os.path.join(cache_dir, f"{mode}_a_mask.npy")
    labels_path = os.path.join(cache_dir, f"{mode}_labels.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(q_ids_path)
        and os.path.exists(q_mask_path)
        and os.path.exists(a_ids_path)
        and os.path.exists(a_mask_path)
    )

    # For train/val, labels must also exist
    if mode in ["train", "val"]:
        cache_exists = cache_exists and os.path.exists(labels_path)

    if load_cached_data and cache_exists:
        print(f"Loading {mode} data from cache...")
        q_ids = np.load(q_ids_path)
        q_mask = np.load(q_mask_path)
        a_ids = np.load(a_ids_path)
        a_mask = np.load(a_mask_path)

        labels = None
        if mode in ["train", "val"]:
            labels = np.load(labels_path)

    else:
        print(f"Processing {mode} data from scratch...")

        # Identify source file
        if mode == "train":
            file_path = Config.TRAIN_PATH
        elif mode == "val":
            file_path = Config.VAL_PATH
        elif mode == "test":
            file_path = Config.TEST_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        df = pd.read_csv(file_path)

        # Debugging subset
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        # Handle text columns
        # Ensure strings
        df["question_title"] = df["question_title"].astype(str).fillna("")
        df["question_body"] = df["question_body"].astype(str).fillna("")
        df["answer"] = df["answer"].astype(str).fillna("")

        # Stream 1: Question (Title + [SEP] + Body)
        # We manually construct the string to ensure the separator is placed correctly
        sep = tokenizer.sep_token if tokenizer.sep_token else " [SEP] "
        q_texts = (df["question_title"] + sep + df["question_body"]).tolist()

        # Stream 2: Answer
        a_texts = df["answer"].tolist()

        # Tokenize
        q_ids, q_mask = tokenize_stream(q_texts, tokenizer, Config.MAX_LEN)
        a_ids, a_mask = tokenize_stream(a_texts, tokenizer, Config.MAX_LEN)

        # Save to cache
        np.save(q_ids_path, q_ids)
        np.save(q_mask_path, q_mask)
        np.save(a_ids_path, a_ids)
        np.save(a_mask_path, a_mask)

        labels = None
        if mode in ["train", "val"]:
            # Extract targets
            # Ensure we only get the columns we expect
            target_cols = [c for c in Config.TARGET_COLS if c in df.columns]
            if len(target_cols) != Config.NUM_LABELS:
                # If columns are missing (shouldn't happen in train/val based on metadata), fill 0 or raise error
                # Based on metadata check, they exist.
                pass

            labels = df[Config.TARGET_COLS].values.astype(np.float32)
            np.save(labels_path, labels)

    return QuestDataset(q_ids, q_mask, a_ids, a_mask, labels)


def get_dataloader(
    mode, tokenizer, batch_size=None, shuffle=None, load_cached_data=True
):
    """
    Factory function to create a DataLoader for a specific mode.
    """
    dataset = get_dataset(mode, tokenizer, load_cached_data=load_cached_data)

    if batch_size is None:
        batch_size = (
            Config.TRAIN_BATCH_SIZE if mode == "train" else Config.VALID_BATCH_SIZE
        )

    if shuffle is None:
        shuffle = True if mode == "train" else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
