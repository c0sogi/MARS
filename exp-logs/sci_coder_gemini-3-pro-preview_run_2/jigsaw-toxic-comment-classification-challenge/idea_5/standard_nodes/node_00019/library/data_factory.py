import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ToxicityDataset(Dataset):
    """
    Custom Dataset for Toxicity Classification.
    Handles tokenization and label formatting.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            tokenizer: Hugging Face tokenizer instance.
            max_len (int): Maximum sequence length for tokenization.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.texts = df["comment_text"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            self.labels = df[Config.LABEL_COLS].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        # Basic whitespace cleanup
        text = " ".join(text.split())

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten tensors to remove the batch dimension added by encode_plus
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # Include token_type_ids if the tokenizer returns them (e.g., BERT/DeBERTa)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].flatten()

        if not self.is_test:
            # Convert labels to float tensor for BCEWithLogitsLoss
            labels = torch.tensor(self.labels[index], dtype=torch.float)
            item["labels"] = labels

        return item


def get_dataloaders(
    tokenizer,
    train_batch_size=Config.TRAIN_BATCH_SIZE,
    valid_batch_size=Config.VALID_BATCH_SIZE,
    max_len=Config.MAX_LEN,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        train_batch_size (int): Batch size for training.
        valid_batch_size (int): Batch size for validation/testing.
        max_len (int): Maximum sequence length.
        debug (bool): Whether to run in debug mode (subset of data).
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames from metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Handle Debug Mode
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = ToxicityDataset(train_df, tokenizer, max_len, is_test=False)
    val_dataset = ToxicityDataset(val_df, tokenizer, max_len, is_test=False)
    test_dataset = ToxicityDataset(test_df, tokenizer, max_len, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
