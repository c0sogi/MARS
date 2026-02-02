import os
import ast
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def clean_text(text):
    """
    Cleans the text by removing surrounding quotes and unescaping unicode characters.
    Handles the specific format of the dataset where text is often a python-style string literal.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Attempt to use literal_eval to handle python-style string escaping and quotes
    # This is effective for strings like '"You are an idiot."' -> 'You are an idiot.'
    try:
        if text.startswith('"') and text.endswith('"'):
            cleaned = ast.literal_eval(text)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback cleanup if literal_eval fails
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except:
        pass

    return text


class InsultDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): Dictionary containing 'input_ids', 'attention_mask', and 'labels' lists.
        """
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def process_split(split_name, file_path, tokenizer, max_len, load_cached_data=True):
    """
    Processes a specific data split (train/val/test).
    Tokenizes text and caches the result to a parquet file to speed up subsequent runs.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{split_name}_tokens.parquet")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return {
                "input_ids": df["input_ids"].tolist(),
                "attention_mask": df["attention_mask"].tolist(),
                "labels": df["labels"].tolist(),
            }
        except Exception:
            # If loading fails (corrupt file), proceed to process from scratch
            pass

    # 2. IF loading fails OR load_cached_data is False: Compute from scratch.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for _, row in df.iterrows():
        text = clean_text(row["Comment"])
        # Handle label: Test set might have placeholder or real labels. Default to 0 if missing/NaN.
        label = row["Insult"] if "Insult" in row and not pd.isna(row["Insult"]) else 0

        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )

        # Convert numpy arrays to lists for storage compatibility
        input_ids_list.append(encoded["input_ids"][0].tolist())
        attention_mask_list.append(encoded["attention_mask"][0].tolist())
        labels_list.append(float(label))

    # Save the result to the cache directory
    df_cache = pd.DataFrame(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    )
    df_cache.to_parquet(cache_path, index=False)

    # 3. Return the data.
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Generates PyTorch DataLoaders for train, validation, and test sets.
    """
    # Train Loader
    train_data = process_split(
        "train", Config.TRAIN_FILE, tokenizer, Config.MAX_LEN, load_cached_data
    )
    train_dataset = InsultDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation Loader
    val_data = process_split(
        "val", Config.VAL_FILE, tokenizer, Config.MAX_LEN, load_cached_data
    )
    val_dataset = InsultDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test Loader
    test_data = process_split(
        "test", Config.TEST_FILE, tokenizer, Config.MAX_LEN, load_cached_data
    )
    test_dataset = InsultDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
