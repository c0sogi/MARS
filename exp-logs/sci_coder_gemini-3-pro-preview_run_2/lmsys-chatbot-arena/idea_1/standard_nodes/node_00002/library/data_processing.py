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


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for Chatbot Preference Prediction.
    Tokenizes text on-the-fly for the transformer.
    """

    def __init__(self, df, tokenizer, max_length=Config.MAX_LENGTH, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract columns to lists for speed
        self.prompts = df["prompt"].fillna("").astype(str).tolist()
        self.res_a = df["response_a"].fillna("").astype(str).tolist()
        self.res_b = df["response_b"].fillna("").astype(str).tolist()

        if not self.is_test:
            # 0: Model A, 1: Model B, 2: Tie
            self.targets = np.argmax(
                df[["winner_model_a", "winner_model_b", "winner_tie"]].values, axis=1
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        res_a = self.res_a[idx]
        res_b = self.res_b[idx]

        # Construct input text: "Prompt [SEP] Response A [SEP] Response B"
        sep = self.tokenizer.sep_token
        full_text = f"{prompt} {sep} {res_a} {sep} {res_b}"

        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.long)

        return item


def create_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    """
    logger.info("Starting data processing...")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Debug Mode: Subsample
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode enabled. Subsampling to {Config.DEBUG_SAMPLE_SIZE} rows."
        )
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Initialize Tokenizer
    logger.info(f"Loading Tokenizer: {Config.TRANSFORMER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)

    # 3. Create Datasets
    train_dataset = ChatbotDataset(train_df, tokenizer)
    val_dataset = ChatbotDataset(val_df, tokenizer)
    test_dataset = ChatbotDataset(test_df, tokenizer, is_test=True)

    # 4. Create DataLoaders
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
