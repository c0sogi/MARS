import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class PhraseDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

        # Pre-fetch columns to lists for faster indexing
        self.ids = df["id"].tolist()
        self.anchors = df["anchor"].astype(str).tolist()
        self.targets = df["target"].astype(str).tolist()
        self.contexts = df["context"].astype(str).tolist()

        if "score" in df.columns:
            self.scores = df["score"].astype(float).tolist()
        else:
            self.scores = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Cite solution_lesson_node_00001: Using pre-trained model input format
        # Construct input text: anchor + [SEP] + target + [SEP] + context
        text = f"{anchor} {self.tokenizer.sep_token} {target} {self.tokenizer.sep_token} {context}"

        encoded = self.tokenizer(
            text,
            max_length=Config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "id": self.ids[idx],
        }

        if self.scores is not None:
            item["score"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item


def collate_fn(batch):
    """
    Custom collate function to stack tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    ids = [item["id"] for item in batch]

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "id": ids,
    }

    if "score" in batch[0]:
        scores = torch.tensor([item["score"] for item in batch], dtype=torch.float)
        batch_out["score"] = scores

    return batch_out


def get_dataloaders(load_cached_data=True):
    """
    Prepares and returns DataLoaders for train, validation, and test sets.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load DataFrames
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Handle Debug Mode
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG Mode: Reduced train size to {len(df_train)}")

    # 2. Initialize Tokenizer
    print(f"Loading tokenizer for {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 3. Create Datasets
    train_dataset = PhraseDataset(df_train, tokenizer)
    val_dataset = PhraseDataset(df_val, tokenizer)
    test_dataset = PhraseDataset(df_test, tokenizer)

    # 4. Create DataLoaders
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
