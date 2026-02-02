import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Prevent tokenizer parallelism issues with DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AuthorDataset(Dataset):
    """
    Custom Dataset for Author Identification.
    Handles tokenization and label encoding.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'text' and optionally 'author' columns.
            tokenizer: Transformers tokenizer instance.
            max_len (int): Maximum sequence length.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.texts = df["text"].fillna("").astype(str).values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Label mapping: EAP -> 0, HPL -> 1, MWS -> 2
        # This order corresponds to the sample_submission columns.
        self.label_map = {"EAP": 0, "HPL": 1, "MWS": 2}

        if not self.is_test:
            self.labels = df["author"].map(self.label_map).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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

        # Flatten to remove batch dimension added by tokenizer
        ids = encoding["input_ids"].flatten()
        mask = encoding["attention_mask"].flatten()

        item = {"input_ids": ids, "attention_mask": mask}

        if not self.is_test:
            # Return label as a long tensor
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            item["labels"] = label

        return item


def get_dataloaders(debug=False):
    """
    Loads data from metadata CSVs, initializes tokenizer, and creates DataLoaders.

    Args:
        debug (bool): If True, subsamples the data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Load Data
    print("Loading metadata CSVs...")
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_META_PATH}")

    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Debugging: Subsample if requested
    if debug or Config.DEBUG:
        print(f"Debug mode active: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # 2. Initialize Tokenizer
    print(f"Initializing tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 3. Create Datasets
    train_dataset = AuthorDataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)

    val_dataset = AuthorDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)

    test_dataset = AuthorDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # 4. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain stability with Grad Accum
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer
