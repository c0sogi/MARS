import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_processing")


class TokenizedDataset(Dataset):
    """
    Dataset that holds pre-tokenized tensors.
    """

    def __init__(self, encodings, targets=None):
        self.encodings = encodings
        self.targets = targets

    def __len__(self):
        return len(self.encodings["prompt_input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.targets is not None:
            return item, torch.tensor(self.targets[idx], dtype=torch.long)
        return item


def tokenize_data(df, tokenizer):
    """
    Tokenizes Prompt, ResA, and ResB columns.
    """

    # Helper to tokenize a list of texts
    def tokenize_list(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
        )

    # Handle NaNs
    prompts = df["prompt"].fillna("").astype(str).tolist()
    res_a = df["response_a"].fillna("").astype(str).tolist()
    res_b = df["response_b"].fillna("").astype(str).tolist()

    logger.info("Tokenizing prompts...")
    p_enc = tokenize_list(prompts)
    logger.info("Tokenizing response A...")
    a_enc = tokenize_list(res_a)
    logger.info("Tokenizing response B...")
    b_enc = tokenize_list(res_b)

    return {
        "prompt_input_ids": p_enc["input_ids"],
        "prompt_attention_mask": p_enc["attention_mask"],
        "res_a_input_ids": a_enc["input_ids"],
        "res_a_attention_mask": a_enc["attention_mask"],
        "res_b_input_ids": b_enc["input_ids"],
        "res_b_attention_mask": b_enc["attention_mask"],
    }


def create_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=False):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    Note: load_cached_data is ignored as we now fine-tune and tokenize on the fly/in-memory.
    """
    logger.info("Starting data processing (Fine-tuning Bi-Encoder)...")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Debug Mode
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode enabled. Subsampling to {Config.DEBUG_SAMPLE_SIZE} rows."
        )
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)

    # 3. Prepare Datasets
    # Train
    train_enc = tokenize_data(train_df, tokenizer)
    train_targets = np.argmax(
        train_df[["winner_model_a", "winner_model_b", "winner_tie"]].values, axis=1
    )
    train_dataset = TokenizedDataset(train_enc, train_targets)

    # Val
    val_enc = tokenize_data(val_df, tokenizer)
    val_targets = np.argmax(
        val_df[["winner_model_a", "winner_model_b", "winner_tie"]].values, axis=1
    )
    val_dataset = TokenizedDataset(val_enc, val_targets)

    # Test
    test_enc = tokenize_data(test_df, tokenizer)
    test_dataset = TokenizedDataset(test_enc, targets=None)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader
