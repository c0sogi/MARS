import os
import re
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification with Stochastic Identity Masking.
    """

    def __init__(self, df, tokenizer, is_train=False):
        self.texts = df[Config.TEXT_COL].astype(str).values

        # Handle targets: Test set might not have targets
        if Config.TARGET_COL in df.columns:
            # Main target
            main_target = df[Config.TARGET_COL].values.reshape(-1, 1)

            # Aux targets
            # Check if aux columns exist (Train/Val), otherwise pad with zeros
            if set(Config.AUX_COLUMNS).issubset(df.columns):
                aux_targets = df[Config.AUX_COLUMNS].fillna(0.0).values
            else:
                aux_targets = np.zeros((len(df), len(Config.AUX_COLUMNS)))

            self.targets = np.hstack([main_target, aux_targets]).astype(np.float32)
        else:
            # Test set (no targets)
            self.targets = np.zeros(
                (len(df), 1 + len(Config.AUX_COLUMNS)), dtype=np.float32
            )

        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_len = Config.MAX_LEN

        # Pre-compile regex for identity keywords if training
        if self.is_train:
            # Create a pattern that matches full words only, case-insensitive
            pattern_str = (
                r"(?i)\b(" + "|".join(map(re.escape, Config.IDENTITY_KEYWORDS)) + r")\b"
            )
            self.identity_pattern = re.compile(pattern_str)
            self.mask_token = self.tokenizer.mask_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Stochastic Identity Masking
        if self.is_train and Config.MASK_PROB > 0:

            def replace_func(match):
                if random.random() < Config.MASK_PROB:
                    return self.mask_token
                return match.group(0)

            text = self.identity_pattern.sub(replace_func, text)

        # Tokenize
        # We truncate here but do NOT pad. Padding is handled in collate_fn for dynamic padding.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,  # Dynamic padding in collator
            return_attention_mask=True,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "target": self.targets[idx],
        }


class DynamicPaddingCollator:
    """
    Collator that pads the batch to the length of the longest sequence in that batch.
    """

    def __init__(self, tokenizer_pad_token_id):
        self.pad_token_id = tokenizer_pad_token_id

    def __call__(self, batch):
        # Find max length in this batch
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids_batch = []
        attention_mask_batch = []
        targets_batch = []

        for x in batch:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]

            # Calculate padding length
            pad_len = max_len - len(input_ids)

            # Pad input_ids
            padded_input_ids = input_ids + [self.pad_token_id] * pad_len
            # Pad attention_mask (0 for padded positions)
            padded_attention_mask = attention_mask + [0] * pad_len

            input_ids_batch.append(padded_input_ids)
            attention_mask_batch.append(padded_attention_mask)
            targets_batch.append(x["target"])

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "target": torch.tensor(targets_batch, dtype=torch.float),
        }


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Data
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # 2. Debug Mode
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # 4. Create Datasets
    train_dataset = ToxicityDataset(train_df, tokenizer, is_train=True)
    val_dataset = ToxicityDataset(val_df, tokenizer, is_train=False)
    test_dataset = ToxicityDataset(test_df, tokenizer, is_train=False)

    # 5. Initialize Collator
    collator = DynamicPaddingCollator(tokenizer.pad_token_id)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,  # Must be False to align with identity_df in evaluation
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
