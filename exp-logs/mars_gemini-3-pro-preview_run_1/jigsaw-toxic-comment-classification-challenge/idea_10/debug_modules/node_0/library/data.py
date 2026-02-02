import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


def get_tokenizer():
    """
    Loads the tokenizer for the model specified in Config.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    return tokenizer


def _load_data(split, load_cached_data=True):
    """
    Loads data for a specific split (train, val, test).
    Merges metadata with raw text and handles caching.
    """
    cache_path = os.path.join(Config.working_dir, f"{split}_cache.parquet")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data from scratch...")

    # Load metadata
    meta_path = os.path.join(Config.metadata_dir, f"{split}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Identify unique source files referenced in metadata
    sources = meta_df["source_file"].unique()
    loaded_frames = []

    for src in sources:
        src_path = os.path.join(Config.input_dir, src)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Raw source file not found: {src_path}")

        raw_df = pd.read_csv(src_path)

        # Filter metadata for this source
        subset_meta = meta_df[meta_df["source_file"] == src]

        # Merge to get text content.
        # Metadata has 'id' and labels. Raw data has 'id' and 'comment_text'.
        merged = pd.merge(
            subset_meta, raw_df[["id", "comment_text"]], on="id", how="left"
        )
        loaded_frames.append(merged)

    # Combine all parts
    df = pd.concat(loaded_frames, ignore_index=True)

    # Handle missing text
    df["comment_text"] = df["comment_text"].fillna("").astype(str)

    # Save to cache
    os.makedirs(Config.working_dir, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Saved {split} data to cache: {cache_path}")

    return df


class ToxicDataset(Dataset):
    """
    Dataset for supervised training, validation, and testing.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.texts = df["comment_text"].values

        if not self.is_test:
            self.labels = df[Config.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-training (Masked Language Modeling).
    Handles dynamic masking of tokens.
    """

    def __init__(self, texts, tokenizer, max_length, mlm_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels for MLM (copy of input_ids)
        labels = input_ids.clone()

        # Probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Mask special tokens (we don't want to mask CLS, SEP, PAD)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Also ensure padding is not masked
        if self.tokenizer.pad_token_id is not None:
            probability_matrix.masked_fill_(
                input_ids == self.tokenizer.pad_token_id, value=0.0
            )

        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for unmasked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The remaining 10% are kept original (but label is still set, so loss is calculated)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_mlm_loader(tokenizer, load_cached_data=True):
    """
    Creates a DataLoader for DAPT using combined train and test text.
    """
    # Load both train and test data
    train_df = _load_data("train", load_cached_data)
    test_df = _load_data("test", load_cached_data)

    # Combine texts
    all_texts = np.concatenate(
        [train_df["comment_text"].values, test_df["comment_text"].values]
    )

    if Config.debug:
        all_texts = all_texts[:1000]

    dataset = MLMDataset(
        texts=all_texts,
        tokenizer=tokenizer,
        max_length=Config.max_length,
        mlm_probability=Config.dapt_mlm_probability,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.dapt_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return loader


def get_train_val_loaders(tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for supervised training and validation.
    """
    train_df = _load_data("train", load_cached_data)
    val_df = _load_data("val", load_cached_data)

    if Config.debug:
        train_df = train_df.head(1000)
        val_df = val_df.head(1000)

    train_dataset = ToxicDataset(train_df, tokenizer, Config.max_length, is_test=False)

    val_dataset = ToxicDataset(val_df, tokenizer, Config.max_length, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(tokenizer, load_cached_data=True):
    """
    Creates a DataLoader for inference on the test set.
    """
    test_df = _load_data("test", load_cached_data)

    if Config.debug:
        test_df = test_df.head(1000)

    test_dataset = ToxicDataset(test_df, tokenizer, Config.max_length, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader, test_df["id"].values
