import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
import nltk
import logging

# Import Config and utils from the provided library files
from library.config import Config
from library.utils import get_logger, seed_everything

# Initialize logger
logger = get_logger("DataModule")

# Ensure nltk resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class FeatureEngineer:
    """
    Extracts explicit meta-features from essay text.
    Features: char_count, word_count, sentence_count, avg_word_length, unique_word_count.
    """

    def __init__(self):
        pass

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting meta-features...")

        # Ensure full_text is string and handle NaNs if any
        texts = df["full_text"].fillna("").astype(str)

        # Character Count
        df["char_count"] = texts.apply(len)

        # Word Count (simple split)
        df["word_count"] = texts.apply(lambda x: len(x.split()))

        # Sentence Count (using nltk for better accuracy than simple split)
        df["sentence_count"] = texts.apply(lambda x: len(nltk.sent_tokenize(x)))

        # Average Word Length
        # Avoid division by zero
        df["avg_word_length"] = df.apply(
            lambda row: (
                row["char_count"] / row["word_count"] if row["word_count"] > 0 else 0
            ),
            axis=1,
        )

        # Unique Word Count
        df["unique_word_count"] = texts.apply(lambda x: len(set(x.split())))

        return df


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Supervised Fine-Tuning (Level 1).
    Returns tokenized inputs and scores.
    """

    def __init__(self, df, tokenizer, max_length=1024, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.texts = df["full_text"].values

        if not self.is_test:
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)

        item = {
            "input_ids": ids,
            "attention_mask": mask,
        }

        if not self.is_test:
            # Convert score to float for regression (MSE Loss)
            score = torch.tensor(self.scores[idx], dtype=torch.float)
            item["labels"] = score

        return item


class MLMDataset(Dataset):
    """
    PyTorch Dataset for Domain-Adaptive Pre-training (Masked Language Modeling).
    Returns masked inputs and labels.
    """

    def __init__(self, texts, tokenizer, max_length=1024, mlm_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels for MLM (clone input_ids)
        labels = input_ids.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Mask special tokens (0 probability)
        # We need to get special tokens mask for the current input
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Determine which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels for unmasked tokens to -100 (ignored by loss function)
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with tokenizer.mask_token_id
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The remaining 10% of the time, keep the original word (but label is still set to the token ID)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_tokenizer():
    """
    Loads the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.model_name)


def make_folds(df: pd.DataFrame, num_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Creates Stratified K-Folds based on the 'score' column.
    """
    df["fold"] = -1
    # StratifiedKFold requires discrete classes.
    # Since score is integer 1-6, it works directly.
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Stratify by score
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["score"])):
        df.loc[val_idx, "fold"] = fold

    return df


def preprocess_data(load_cached_data: bool = True):
    """
    Main data processing pipeline.
    Loads raw metadata, performs feature engineering, creates folds, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, test_df)
    """
    seed_everything(Config.seed)

    train_cache_path = os.path.join(Config.cache_dir, "train_processed.parquet")
    test_cache_path = os.path.join(Config.cache_dir, "test_processed.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            logger.info(f"Loading cached data from {Config.cache_dir}")
            try:
                train_df = pd.read_parquet(train_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, test_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-processing...")
        else:
            logger.info("Cache not found. Processing from scratch...")
    else:
        logger.info("Ignoring cache. Processing from scratch...")

    # Load Metadata
    logger.info(f"Loading raw data from {Config.metadata_dir}")
    train_df = pd.read_csv(Config.train_path)
    test_df = pd.read_csv(Config.test_path)

    # Feature Engineering
    fe = FeatureEngineer()
    train_df = fe.extract_features(train_df)
    test_df = fe.extract_features(test_df)

    # Create Folds
    logger.info(f"Creating {Config.num_folds} stratified folds...")
    train_df = make_folds(train_df, num_folds=Config.num_folds, seed=Config.seed)

    # Save to Cache
    logger.info(f"Saving processed data to {Config.cache_dir}")
    os.makedirs(Config.cache_dir, exist_ok=True)
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df
