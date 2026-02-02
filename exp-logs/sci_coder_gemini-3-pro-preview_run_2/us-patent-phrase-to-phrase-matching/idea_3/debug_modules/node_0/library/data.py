import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_cpc_texts


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for the Phrase Similarity Task.
    Handles tokenization and label conversion for the classification head.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-fetch columns to numpy arrays for efficiency
        self.texts = df["input_text"].values
        self.ids = df["id"].values

        if not self.is_test:
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize the input text
        # The text is already formatted as: "Context Description [SEP] Anchor [SEP] Target"
        # The tokenizer will add [CLS] at the start and [SEP] at the end.
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove the batch dimension added by tokenizer
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ids": self.ids[idx],
        }

        if not self.is_test:
            score = self.scores[idx]
            # Convert continuous score to class index for 5-class classification
            # 0.00 -> 0, 0.25 -> 1, 0.50 -> 2, 0.75 -> 3, 1.00 -> 4
            label = int(round(score * 4))
            item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


def _process_data(file_path, cpc_texts, tokenizer_sep, cache_path, load_cached_data):
    """
    Loads raw data, performs context enrichment, constructs input text, and handles caching.
    """
    # 1. Check and load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If cache load fails, proceed to process from scratch
            pass

    # 2. Load raw data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Handle Debug Mode
    if Config.debug:
        df = df.head(Config.debug_sample_size).copy()

    # 3. Context Enrichment and Text Construction
    # We map the CPC code to its description.
    # Format: Context Description + [SEP] + Anchor + [SEP] + Target
    def construct_input(row):
        code = row.get("context", "")
        # Retrieve description, fallback to code if not found
        context_desc = cpc_texts.get(code, code)
        anchor = row.get("anchor", "")
        target = row.get("target", "")

        # Ensure strings
        context_desc = str(context_desc).strip()
        anchor = str(anchor).strip()
        target = str(target).strip()

        return f"{context_desc}{tokenizer_sep}{anchor}{tokenizer_sep}{target}"

    df["input_text"] = df.apply(construct_input, axis=1)

    # 4. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def prepare_loaders(tokenizer, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        tokenizer: The HuggingFace tokenizer instance.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load CPC Context Descriptions
    cpc_texts = get_cpc_texts(Config.cpc_codes_path)

    # Determine cache filenames (append _debug if in debug mode)
    suffix = "_debug" if Config.debug else ""
    train_cache = os.path.join(Config.output_dir, f"train_processed{suffix}.parquet")
    val_cache = os.path.join(Config.output_dir, f"val_processed{suffix}.parquet")
    test_cache = os.path.join(Config.output_dir, f"test_processed{suffix}.parquet")

    # Process Data
    train_df = _process_data(
        Config.train_path, cpc_texts, tokenizer.sep_token, train_cache, load_cached_data
    )

    val_df = _process_data(
        Config.val_path, cpc_texts, tokenizer.sep_token, val_cache, load_cached_data
    )

    test_df = _process_data(
        Config.test_path, cpc_texts, tokenizer.sep_token, test_cache, load_cached_data
    )

    # Initialize Datasets
    train_ds = PhraseDataset(train_df, tokenizer, Config.max_length, is_test=False)
    val_ds = PhraseDataset(val_df, tokenizer, Config.max_length, is_test=False)
    test_ds = PhraseDataset(test_df, tokenizer, Config.max_length, is_test=True)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
