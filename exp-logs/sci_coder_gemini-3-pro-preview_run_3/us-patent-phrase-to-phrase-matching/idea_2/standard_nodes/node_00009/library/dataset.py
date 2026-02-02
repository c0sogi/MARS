import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from library.config import Config


def load_and_process_data(input_path, cache_path, load_cached_data=True):
    """
    Loads data from CSV or Parquet cache.
    Strictly follows the caching logic required.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from source."
            )

    # 2. Load from source if cache failed or not requested
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Ensure text columns are strings
    text_cols = ["anchor", "target", "context"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


class PhraseDataset(Dataset):
    """
    Dataset for Cross-Encoder Phrase Matching.
    Concatenates context, anchor, and target into a single input sequence.
    """

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = tokenizer.sep_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        context = row["context"]
        anchor = row["anchor"]
        target = row["target"]

        # Construct input string: [CLS] context [SEP] anchor [SEP] target [SEP]
        # The tokenizer adds [CLS] and the final [SEP] automatically.
        # We manually insert separators between the segments.
        text = f"{context} {self.sep_token} {anchor} {self.sep_token} {target}"

        # Tokenize without padding (padding is handled in collator)
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,  # Return python lists
        )

        item = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        # Add label if available (regression target)
        if "score" in row:
            item["labels"] = float(row["score"])

        # Add ID for inference tracking
        if "id" in row:
            item["id"] = row["id"]

        return item


class CustomCollator:
    """
    Collator that handles dynamic padding for inputs and stacking for regression labels.
    """

    def __init__(self, tokenizer):
        self.hf_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, batch):
        # Separate inputs (handled by HF collator) from labels/ids
        input_batch = []
        labels = []
        ids = []

        has_labels = "labels" in batch[0]
        has_ids = "id" in batch[0]

        for item in batch:
            # Extract only tokenizer outputs for the HF collator
            input_item = {
                k: v
                for k, v in item.items()
                if k in ["input_ids", "attention_mask", "token_type_ids"]
            }
            input_batch.append(input_item)

            if has_labels:
                labels.append(item["labels"])
            if has_ids:
                ids.append(item["id"])

        # Dynamic padding of inputs
        batch_out = self.hf_collator(input_batch)

        # Stack labels
        if has_labels:
            batch_out["labels"] = torch.tensor(labels, dtype=torch.float)

        # Pass IDs through (list of strings)
        if has_ids:
            batch_out["id"] = ids

        return batch_out


def get_dataloaders(tokenizer, load_cached_data=True, debug=False):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # 1. Load Data
    train_df = load_and_process_data(
        Config.train_path, Config.train_cache_path, load_cached_data
    )
    val_df = load_and_process_data(
        Config.val_path, Config.val_cache_path, load_cached_data
    )
    test_df = load_and_process_data(
        Config.test_path, Config.test_cache_path, load_cached_data
    )

    # Debugging: Reduce dataset size if requested
    if debug or Config.debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # 2. Create Datasets
    train_dataset = PhraseDataset(train_df, tokenizer, Config.max_length)
    val_dataset = PhraseDataset(val_df, tokenizer, Config.max_length)
    test_dataset = PhraseDataset(test_df, tokenizer, Config.max_length)

    # 3. Initialize Collator
    collator = CustomCollator(tokenizer)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
