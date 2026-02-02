import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class QuestDataset(Dataset):
    """
    PyTorch Dataset for the Causal-Aware Siamese DeBERTa Network.
    Handles dual-stream input (Question and Answer) and target labels.
    """

    def __init__(self, data, is_test=False):
        """
        Args:
            data (dict): Dictionary containing numpy arrays for inputs and labels.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.data = data
        self.is_test = is_test

        # Unpack common keys for easier access (assuming they exist)
        self.q_input_ids = data["q_input_ids"]
        self.q_attention_mask = data["q_attention_mask"]
        self.a_input_ids = data["a_input_ids"]
        self.a_attention_mask = data["a_attention_mask"]

        # Optional token_type_ids (if produced by tokenizer)
        self.q_token_type_ids = data.get("q_token_type_ids", None)
        self.a_token_type_ids = data.get("a_token_type_ids", None)

        if not self.is_test:
            self.labels = data["labels"]

    def __len__(self):
        return len(self.q_input_ids)

    def __getitem__(self, idx):
        # Prepare inputs for the Question Head
        item = {
            "input_ids_q": torch.tensor(self.q_input_ids[idx], dtype=torch.long),
            "attention_mask_q": torch.tensor(
                self.q_attention_mask[idx], dtype=torch.long
            ),
            "input_ids_a": torch.tensor(self.a_input_ids[idx], dtype=torch.long),
            "attention_mask_a": torch.tensor(
                self.a_attention_mask[idx], dtype=torch.long
            ),
        }

        # Add token_type_ids if they exist (DeBERTa V3 uses them for pairs)
        if self.q_token_type_ids is not None:
            item["token_type_ids_q"] = torch.tensor(
                self.q_token_type_ids[idx], dtype=torch.long
            )
        if self.a_token_type_ids is not None:
            item["token_type_ids_a"] = torch.tensor(
                self.a_token_type_ids[idx], dtype=torch.long
            )

        # Add labels for training/validation
        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        # Add qa_id for tracking/submission (optional, but good for debugging)
        if "qa_ids" in self.data:
            item["qa_id"] = self.data["qa_ids"][idx]

        return item


def _process_split(split_name, file_path, tokenizer, cfg, load_cached_data):
    """
    Internal helper to load, tokenize, and cache data for a specific split.
    """
    # Ensure cache directory exists
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)

    # Define cache filenames
    # We append _debug if running in debug mode to avoid polluting full cache
    suffix = "_debug" if cfg.DEBUG else ""

    cache_files = {
        "q_input_ids": os.path.join(
            cfg.WORKING_DIR, f"{split_name}_q_input_ids{suffix}.npy"
        ),
        "q_attention_mask": os.path.join(
            cfg.WORKING_DIR, f"{split_name}_q_attention_mask{suffix}.npy"
        ),
        "q_token_type_ids": os.path.join(
            cfg.WORKING_DIR, f"{split_name}_q_token_type_ids{suffix}.npy"
        ),
        "a_input_ids": os.path.join(
            cfg.WORKING_DIR, f"{split_name}_a_input_ids{suffix}.npy"
        ),
        "a_attention_mask": os.path.join(
            cfg.WORKING_DIR, f"{split_name}_a_attention_mask{suffix}.npy"
        ),
        "a_token_type_ids": os.path.join(
            cfg.WORKING_DIR, f"{split_name}_a_token_type_ids{suffix}.npy"
        ),
        "qa_ids": os.path.join(cfg.WORKING_DIR, f"{split_name}_qa_ids{suffix}.npy"),
    }

    if split_name != "test":
        cache_files["labels"] = os.path.join(
            cfg.WORKING_DIR, f"{split_name}_labels{suffix}.npy"
        )

    # Check if cache exists
    # We only require input_ids and masks to exist. token_type_ids are optional depending on tokenizer.
    required_keys = [
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "qa_ids",
    ]
    if split_name != "test":
        required_keys.append("labels")

    cache_exists = all(os.path.exists(cache_files[k]) for k in required_keys)

    # 1. Load from Cache
    if load_cached_data and cache_exists:
        print(f"Loading cached data for '{split_name}' split...")
        data = {}
        for k, path in cache_files.items():
            if os.path.exists(path):
                data[k] = np.load(path)
        return data

    # 2. Process from Scratch
    print(f"Processing data for '{split_name}' split from source...")
    df = pd.read_csv(file_path)

    # Handle Debug Mode
    if cfg.DEBUG:
        print(f"DEBUG mode: processing only {cfg.DEBUG_SAMPLES} samples.")
        df = df.iloc[: cfg.DEBUG_SAMPLES].copy()

    # Text Cleaning
    df["question_title"] = df["question_title"].fillna("").astype(str)
    df["question_body"] = df["question_body"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    # Tokenize Question Stream (Title + [SEP] + Body)
    # DeBERTa tokenizer handles pairs automatically
    q_enc = tokenizer(
        text=df["question_title"].tolist(),
        text_pair=df["question_body"].tolist(),
        max_length=cfg.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_token_type_ids=True,  # Request explicitly, though AutoTokenizer might ignore if not supported
    )

    # Tokenize Answer Stream (Answer only)
    a_enc = tokenizer(
        text=df["answer"].tolist(),
        max_length=cfg.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_token_type_ids=True,
    )

    # Construct Data Dictionary
    data = {
        "q_input_ids": q_enc["input_ids"],
        "q_attention_mask": q_enc["attention_mask"],
        "a_input_ids": a_enc["input_ids"],
        "a_attention_mask": a_enc["attention_mask"],
        "qa_ids": df["qa_id"].values,
    }

    # Add token_type_ids if generated
    if "token_type_ids" in q_enc:
        data["q_token_type_ids"] = q_enc["token_type_ids"]
    if "token_type_ids" in a_enc:
        data["a_token_type_ids"] = a_enc["token_type_ids"]

    # Process Labels
    if split_name != "test":
        # Ensure we select columns in the exact order defined in Config
        labels = df[cfg.TARGET_COLS].values.astype(np.float32)
        data["labels"] = labels

    # Save to Cache
    print(f"Saving processed data to cache: {cfg.WORKING_DIR}")
    for k, v in data.items():
        if k in cache_files:
            np.save(cache_files[k], v)

    return data


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles caching of tokenized data to speed up subsequent runs.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        load_cached_data (bool): If True, attempts to load pre-processed numpy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    cfg = Config()

    # Process Train
    train_data = _process_split(
        "train", cfg.TRAIN_PATH, tokenizer, cfg, load_cached_data
    )
    train_dataset = QuestDataset(train_data, is_test=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize training
    )

    # Process Validation
    val_data = _process_split("val", cfg.VAL_PATH, tokenizer, cfg, load_cached_data)
    val_dataset = QuestDataset(val_data, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Process Test
    test_data = _process_split("test", cfg.TEST_PATH, tokenizer, cfg, load_cached_data)
    test_dataset = QuestDataset(test_data, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
