import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from library.config import Config
from library.utils import get_logger

logger = get_logger()


class TextCleaner:
    """
    Handles text normalization for the dataset.
    Removes wrapping quotes and handles unicode escapes.
    """

    @staticmethod
    def clean(text):
        if pd.isna(text):
            return ""

        text = str(text)

        # Attempt to use literal_eval to handle python-style string escaping and quotes
        try:
            # If it starts and ends with quotes, it might be a string literal
            if text.startswith('"') and text.endswith('"'):
                # This handles escaped characters like \n, \xe2, etc.
                cleaned = ast.literal_eval(text)
                return cleaned
        except (ValueError, SyntaxError):
            pass

        # Fallback cleanup if literal_eval fails
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Basic unicode unescape if needed
        try:
            text = text.encode("utf-8").decode("unicode_escape")
        except:
            pass

        return text


class InsultDataset(Dataset):
    """
    Dataset for Supervised Classification.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df[Config.input_col].values

        if not self.is_test:
            self.targets = df[Config.target_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            target = self.targets[idx]
            item["target"] = torch.tensor(target, dtype=torch.float)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (TAPT).
    Returns tokenized inputs. Masking is expected to be handled by DataCollator.
    """

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            # Labels are usually created by the DataCollatorForLanguageModeling
            # But we can pass input_ids as labels for the collator to mask
            "labels": encoding["input_ids"].flatten(),
        }


def get_tokenizer():
    """Loads the tokenizer defined in Config."""
    return AutoTokenizer.from_pretrained(Config.model_name)


def prepare_supervised_data(load_cached_data=True):
    """
    Loads train, val, and test data.
    Implements caching mechanism using Parquet.
    """
    # Define cache paths
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_cleaned.parquet")
    val_cache = os.path.join(cache_dir, "val_cleaned.parquet")
    test_cache = os.path.join(cache_dir, "test_cleaned.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            logger.info("Loading cleaned data from cache...")
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)
            return df_train, df_val, df_test
        else:
            logger.info("Cache not found. Processing from scratch...")
    else:
        logger.info("Ignoring cache. Processing from scratch...")

    # 2. Load from metadata
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # 3. Clean text
    logger.info("Cleaning text data...")
    df_train[Config.input_col] = df_train[Config.input_col].apply(TextCleaner.clean)
    df_val[Config.input_col] = df_val[Config.input_col].apply(TextCleaner.clean)
    df_test[Config.input_col] = df_test[Config.input_col].apply(TextCleaner.clean)

    # 4. Save to cache
    logger.info(f"Saving cleaned data to {cache_dir}...")
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    return df_train, df_val, df_test


def prepare_tapt_data(df_train, df_val, df_test):
    """
    Combines all available text for Task-Adaptive Pre-Training.
    """
    texts = pd.concat(
        [
            df_train[Config.input_col],
            df_val[Config.input_col],
            df_test[Config.input_col],
        ]
    ).tolist()

    # Remove empty strings
    texts = [t for t in texts if len(str(t).strip()) > 0]

    logger.info(f"Prepared TAPT corpus with {len(texts)} samples.")
    return texts


def prepare_pseudo_data(df_train, df_test, test_probs):
    """
    Merges high-confidence test predictions into the training set.

    Args:
        df_train: Original training dataframe.
        df_test: Test dataframe.
        test_probs: Numpy array or list of predicted probabilities for the test set.
    """
    logger.info("Preparing pseudo-labeled data...")

    # Assign predictions to test dataframe
    df_pseudo = df_test.copy()
    df_pseudo[Config.target_col] = test_probs

    # Filter high confidence samples
    # Class 1 (Insult) > High Threshold
    high_conf_1 = df_pseudo[
        df_pseudo[Config.target_col] > Config.conf_thresh_high
    ].copy()
    high_conf_1[Config.target_col] = 1

    # Class 0 (Neutral) < Low Threshold
    high_conf_0 = df_pseudo[
        df_pseudo[Config.target_col] < Config.conf_thresh_low
    ].copy()
    high_conf_0[Config.target_col] = 0

    # Combine
    df_pseudo_filtered = pd.concat([high_conf_1, high_conf_0])

    # Ensure types match
    df_pseudo_filtered[Config.target_col] = df_pseudo_filtered[
        Config.target_col
    ].astype(int)

    logger.info(
        f"Selected {len(df_pseudo_filtered)} pseudo-labeled samples "
        f"({len(high_conf_0)} neutral, {len(high_conf_1)} insult)."
    )

    # Merge with original train
    # We only keep necessary columns to avoid schema mismatch if date is missing/different format
    cols = [Config.input_col, Config.target_col]

    # Handle Date column if present in both, otherwise drop
    if "Date" in df_train.columns and "Date" in df_pseudo_filtered.columns:
        cols.append("Date")

    df_augmented = pd.concat([df_train[cols], df_pseudo_filtered[cols]]).reset_index(
        drop=True
    )

    logger.info(f"Augmented training set size: {len(df_augmented)}")

    return df_augmented
