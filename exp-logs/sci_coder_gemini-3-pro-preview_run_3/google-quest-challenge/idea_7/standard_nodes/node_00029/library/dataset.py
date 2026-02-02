import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config
from library.utils import (
    save_numpy_array,
    load_numpy_array,
)


def get_tokenizer(model_name=Config.MODEL_DEBERTA):
    """
    Loads and returns the tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


class MLMDataset(Dataset):
    """
    Dataset for Domain Adaptation (Masked Language Modeling).
    """

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }


def prepare_mlm_data(load_cached_data=True, tokenizer=None):
    """
    Prepares data for MLM by concatenating text from train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to load from cache if available.
        tokenizer: Transformers tokenizer.

    Returns:
        MLMDataset: The dataset object.
    """
    cache_prefix = "mlm_data"
    f_input_ids = f"{cache_prefix}_input_ids.npy"
    f_attn_mask = f"{cache_prefix}_attention_mask.npy"

    # Try loading from cache
    if load_cached_data:
        input_ids = load_numpy_array(f_input_ids)
        attention_mask = load_numpy_array(f_attn_mask)

        if input_ids is not None and attention_mask is not None:
            print(f"Loaded MLM data from cache: {len(input_ids)} samples.")
            return MLMDataset(input_ids, attention_mask)

    print("Processing MLM data from scratch...")
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Concatenate all available text
    text_cols = ["question_title", "question_body", "answer"]
    all_texts = []

    for df in [train_df, val_df, test_df]:
        for col in text_cols:
            if col in df.columns:
                # Fill NaNs and convert to string
                texts = df[col].fillna("").astype(str).tolist()
                all_texts.extend(texts)

    # Filter empty strings
    all_texts = [t for t in all_texts if len(t.strip()) > 0]

    # Tokenize
    print(f"Tokenizing {len(all_texts)} text segments for MLM...")
    encodings = tokenizer(
        all_texts,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Save to cache
    save_numpy_array(input_ids, f_input_ids)
    save_numpy_array(attention_mask, f_attn_mask)

    return MLMDataset(input_ids, attention_mask)


class StackExchangeDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning and Inference.
    Returns input_ids, masks, and targets.
    """

    def __init__(self, input_ids, attention_mask, q_mask, a_mask, targets=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.q_mask = q_mask
        self.a_mask = a_mask
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "q_mask": torch.tensor(self.q_mask[idx], dtype=torch.float),
            "a_mask": torch.tensor(self.a_mask[idx], dtype=torch.float),
        }

        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def prepare_supervised_data(split="train", load_cached_data=True, tokenizer=None):
    """
    Prepares data for supervised training or testing.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache.
        tokenizer: Transformers tokenizer.

    Returns:
        StackExchangeDataset: The dataset object.
    """
    cache_prefix = f"supervised_{split}"

    f_input_ids = f"{cache_prefix}_input_ids.npy"
    f_attn_mask = f"{cache_prefix}_attention_mask.npy"
    f_q_mask = f"{cache_prefix}_q_mask.npy"
    f_a_mask = f"{cache_prefix}_a_mask.npy"
    f_targets = f"{cache_prefix}_targets.npy"

    # Try loading from cache
    if load_cached_data:
        input_ids = load_numpy_array(f_input_ids)
        attention_mask = load_numpy_array(f_attn_mask)
        q_mask = load_numpy_array(f_q_mask)
        a_mask = load_numpy_array(f_a_mask)
        targets = load_numpy_array(f_targets)

        # Check if essential files exist
        if input_ids is not None:
            # If split is test, targets might be None, which is fine
            if split == "test" or targets is not None:
                print(f"Loaded {split} data from cache: {len(input_ids)} samples.")
                return StackExchangeDataset(
                    input_ids, attention_mask, q_mask, a_mask, targets
                )

    print(f"Processing {split} data from scratch...")
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # Load correct dataframe
    if split == "train":
        df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    elif split == "val":
        df = pd.read_csv(Config.VAL_METADATA_PATH)
    elif split == "test":
        df = pd.read_csv(Config.TEST_METADATA_PATH)
    else:
        raise ValueError(f"Unknown split: {split}")

    # Prepare text inputs
    # Format: [CLS] Question Title + Question Body [SEP] Answer [SEP]
    df["question_full"] = (
        df["question_title"].fillna("") + " " + df["question_body"].fillna("")
    )
    df["answer_full"] = df["answer"].fillna("")

    questions = df["question_full"].astype(str).tolist()
    answers = df["answer_full"].astype(str).tolist()

    # Tokenize
    print(f"Tokenizing {len(questions)} QA pairs for {split}...")
    encodings = tokenizer(
        questions,
        answers,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Generate q_mask and a_mask using sequence_ids
    batch_size = len(input_ids)
    seq_len = Config.MAX_LEN
    q_mask = np.zeros((batch_size, seq_len), dtype=np.float32)
    a_mask = np.zeros((batch_size, seq_len), dtype=np.float32)

    for i in range(batch_size):
        # seq_ids is a list of [None, 0, 0, ..., None, 1, 1, ..., None]
        seq_ids = encodings.sequence_ids(i)

        for j, sid in enumerate(seq_ids):
            if sid == 0:
                q_mask[i, j] = 1.0
            elif sid == 1:
                a_mask[i, j] = 1.0

    # Prepare targets
    targets = None
    if split != "test":
        targets = df[Config.TARGET_COLS].values.astype(np.float32)
        save_numpy_array(targets, f_targets)

    # Save features
    save_numpy_array(input_ids, f_input_ids)
    save_numpy_array(attention_mask, f_attn_mask)
    save_numpy_array(q_mask, f_q_mask)
    save_numpy_array(a_mask, f_a_mask)

    return StackExchangeDataset(input_ids, attention_mask, q_mask, a_mask, targets)
