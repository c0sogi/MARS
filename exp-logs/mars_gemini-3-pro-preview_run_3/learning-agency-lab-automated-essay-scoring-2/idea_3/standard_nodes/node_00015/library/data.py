import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


def extract_meta_features(text):
    """
    Extracts structural and lexical features from the text.

    Args:
        text (str): The essay text.

    Returns:
        dict: A dictionary of scalar features.
    """
    # Basic string stats
    char_count = len(text)
    text_stripped = text.strip()

    # Word stats
    words = text.split()
    word_count = len(words)

    # Sentence stats (approximate using punctuation)
    # Split by common sentence terminators
    sentences = re.split(r"[.!?]+", text)
    # Filter out empty strings resulting from split
    sentences = [s for s in sentences if s.strip()]
    sentence_count = len(sentences)
    if sentence_count == 0 and char_count > 0:
        sentence_count = 1

    # Paragraph stats
    paragraph_count = len(text.split("\n\n"))

    # Vocabulary stats
    unique_words = set(words)
    unique_word_count = len(unique_words)

    # Ratios and averages
    avg_word_len = char_count / word_count if word_count > 0 else 0.0
    unique_word_ratio = unique_word_count / word_count if word_count > 0 else 0.0
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0

    return {
        "word_count": float(word_count),
        "char_count": float(char_count),
        "sentence_count": float(sentence_count),
        "paragraph_count": float(paragraph_count),
        "avg_word_len": float(avg_word_len),
        "unique_word_count": float(unique_word_count),
        "unique_word_ratio": float(unique_word_ratio),
        "avg_sentence_len": float(avg_sentence_len),
    }


def process_data(csv_path, split_name, load_cached_data=True):
    """
    Loads metadata, computes meta-features, and handles caching via Parquet.

    Args:
        csv_path (str): Path to the source metadata CSV.
        split_name (str): Name of the split (train, val, test) for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with meta-features.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached processed data from: {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"Processing data for {split_name}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Compute meta-features
    # Using a list of dicts is generally faster than applying row-by-row to append to DF
    features_list = [extract_meta_features(str(text)) for text in df["full_text"]]
    features_df = pd.DataFrame(features_list)

    # Concatenate original data with new features
    df = pd.concat([df, features_df], axis=1)

    # 3. Save to cache
    print(f"Saving processed data to: {cache_file}")
    df.to_parquet(cache_file, index=False)

    return df


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Handles tokenization and meta-feature retrieval.
    """

    def __init__(self, df, tokenizer, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train

        # Pre-extract numpy arrays for faster access
        self.texts = df["full_text"].astype(str).values
        self.essay_ids = df["essay_id"].values

        # Identify meta-feature columns (exclude non-feature cols)
        non_feature_cols = {"essay_id", "full_text", "score", "source_file"}
        self.feature_cols = [c for c in df.columns if c not in non_feature_cols]
        self.meta_features = df[self.feature_cols].values.astype(np.float32)

        if self.is_train:
            self.labels = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        # For backbone training, we use standard truncation to max_length.
        # Sliding window logic is handled separately during inference aggregation.
        encoding = self.tokenizer(
            text,
            max_length=Config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return standard lists
            add_special_tokens=True,
        )

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "meta_features": torch.tensor(self.meta_features[idx], dtype=torch.float32),
            "essay_id": self.essay_ids[idx],
        }

        if self.is_train:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def tokenize_sliding_window(text, tokenizer):
    """
    Helper function to tokenize text into overlapping chunks for inference.

    Args:
        text (str): Input text.
        tokenizer: Transformers tokenizer.

    Returns:
        dict: Dictionary containing stacked tensors for 'input_ids' and 'attention_mask'.
    """
    encoding = tokenizer(
        text,
        max_length=Config.chunk_size,
        stride=Config.stride,
        return_overflowing_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.

    Args:
        tokenizer: Transformers tokenizer instance.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Process Data
    train_df = process_data(Config.TRAIN_META, "train", load_cached_data)
    val_df = process_data(Config.VAL_META, "val", load_cached_data)

    # Create Datasets
    train_dataset = EssayDataset(train_df, tokenizer, is_train=True)
    val_dataset = EssayDataset(val_df, tokenizer, is_train=True)

    # Create DataLoaders
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
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(tokenizer, load_cached_data=True):
    """
    Creates DataLoader for the test set.

    Args:
        tokenizer: Transformers tokenizer instance.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        DataLoader: Test data loader.
    """
    test_df = process_data(Config.TEST_META, "test", load_cached_data)
    test_dataset = EssayDataset(test_df, tokenizer, is_train=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader
