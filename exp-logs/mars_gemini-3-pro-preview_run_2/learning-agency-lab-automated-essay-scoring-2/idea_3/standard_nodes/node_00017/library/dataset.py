import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from library.config import Config

# --- 1. Preprocessing & Caching Logic ---


def preprocess_text(text):
    """
    Applies minimal text cleaning: whitespace normalization.
    This ensures consistency in input text representation.
    """
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def load_data(split, load_cached_data=True):
    """
    Loads data for a specific split (train/val/test).
    Implements caching mechanism using Parquet to store preprocessed DataFrames.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    cache_filename = f"{split}_processed.parquet"
    cache_path = os.path.join(Config.output_dir, cache_filename)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Re-processing.")

    # 2. If not cached or load failed, load raw and process
    if split == "train":
        src_path = Config.train_path
    elif split == "val":
        src_path = Config.val_path
    elif split == "test":
        src_path = Config.test_path
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    df = pd.read_csv(src_path)

    # Apply deterministic processing
    if "full_text" in df.columns:
        df["full_text"] = df["full_text"].apply(preprocess_text)

    # 3. Save to cache
    os.makedirs(Config.output_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


# --- 2. Dataset Class ---


class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, is_test=False):
        """
        PyTorch Dataset for Essay Scoring.

        Args:
            df (pd.DataFrame): DataFrame containing 'full_text' and 'score' (if not test).
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.texts = df["full_text"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Handle scores if not test set
        if not self.is_test:
            if "score" in df.columns:
                self.scores = df["score"].values.astype(float)
            else:
                # Should not happen given metadata integrity checks
                self.scores = [0.0] * len(df)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        # We use truncation but NO padding here. Padding is handled by DataCollator.
        # This saves memory and compute by padding to the batch max length, not global max.
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,  # Return python lists
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            # Return label as a float tensor for regression
            item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return item


# --- 3. DataLoader Factory ---


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        debug (bool): If True, uses a small subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load DataFrames
    train_df = load_data("train", load_cached_data)
    val_df = load_data("val", load_cached_data)
    test_df = load_data("test", load_cached_data)

    # Debug Mode: Subset data
    if debug or Config.debug:
        train_df = train_df.head(Config.debug_subset_size).reset_index(drop=True)
        val_df = val_df.head(Config.debug_subset_size).reset_index(drop=True)
        test_df = test_df.head(Config.debug_subset_size).reset_index(drop=True)

    # Create Datasets
    train_dataset = EssayDataset(
        train_df, tokenizer, max_length=Config.max_length, is_test=False
    )
    val_dataset = EssayDataset(
        val_df, tokenizer, max_length=Config.max_length, is_test=False
    )
    test_dataset = EssayDataset(
        test_df, tokenizer, max_length=Config.max_length, is_test=True
    )

    # Data Collator
    # This automatically pads input_ids and attention_mask to the longest in the batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
