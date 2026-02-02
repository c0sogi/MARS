import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(os.path.join(Config.output_dir, "data.log"))


def get_meta_features(df):
    """
    Calculates explicit structural scalar features for the text data.

    Features:
    - word_count: Total number of words.
    - char_count: Total number of characters.
    - sentence_count: Approximate number of sentences.
    - unique_word_ratio: Ratio of unique words to total words (lexical diversity).

    Args:
        df (pd.DataFrame): DataFrame containing a 'full_text' column.

    Returns:
        pd.DataFrame: DataFrame with added meta-feature columns.
    """
    # Ensure text is string
    texts = df["full_text"].fillna("").astype(str)

    # Word count
    df["word_count"] = texts.apply(lambda x: len(x.split()))

    # Character count
    df["char_count"] = texts.apply(len)

    # Sentence count (simple heuristic using punctuation)
    df["sentence_count"] = texts.apply(lambda x: len(re.findall(r"[.!?]+", x)) + 1)

    # Unique word ratio
    def calculate_unique_ratio(text):
        words = text.split()
        if len(words) == 0:
            return 0.0
        return len(set(words)) / len(words)

    df["unique_word_ratio"] = texts.apply(calculate_unique_ratio)

    return df


def make_folds(df, config):
    """
    Splits the dataset into stratified folds based on the score.

    Args:
        df (pd.DataFrame): DataFrame containing 'score' column.
        config (Config): Configuration object with n_folds and seed.

    Returns:
        pd.DataFrame: DataFrame with a new 'fold' column.
    """
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    df["fold"] = -1
    # Stratify by 'score'
    for fold, (_, val_idx) in enumerate(skf.split(df, df["score"])):
        df.loc[val_idx, "fold"] = fold

    return df


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.

    Handles tokenization and tensor conversion for the DeBERTa backbone.
    """

    def __init__(self, df, tokenizer, config, is_train=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing text and metadata.
            tokenizer: HuggingFace tokenizer.
            config (Config): Configuration object.
            is_train (bool): Whether this is a training dataset (includes labels).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train

        # Pre-extract data to avoid pandas overhead in __getitem__
        self.texts = df["full_text"].values.tolist()
        self.essay_ids = df["essay_id"].values.tolist()

        # Meta features columns
        self.meta_cols = [
            "word_count",
            "char_count",
            "sentence_count",
            "unique_word_ratio",
        ]
        self.meta_features = df[self.meta_cols].values.astype(np.float32)

        if self.is_train:
            self.labels = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenization
        # We use truncation for training stability.
        # Sliding window logic is typically handled in the inference loop or a custom collate
        # if multi-view training is required, but standard fine-tuning uses truncation.
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        meta_feats = torch.tensor(self.meta_features[idx], dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "meta_features": meta_feats,
            "essay_id": self.essay_ids[idx],
        }

        if self.is_train:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def load_and_preprocess(config, load_cached_data=True):
    """
    Loads raw metadata, merges train/val for full CV, calculates meta-features,
    creates folds, and caches the result.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, test_df)
    """
    # Define cache paths
    train_cache_path = os.path.join(config.cache_dir, "train_processed.parquet")
    test_cache_path = os.path.join(config.cache_dir, "test_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            logger.info(f"Loading processed data from cache: {config.cache_dir}")
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)

            # If debug mode, we might need to slice again if the cached data is full size
            # But usually we cache the specific debug state.
            # To be safe, if config.debug is True, we re-slice.
            if config.debug:
                train_df = train_df.iloc[:100].reset_index(drop=True)
                test_df = test_df.iloc[:100].reset_index(drop=True)

            return train_df, test_df
        else:
            logger.info("Cache not found or incomplete. Processing from scratch...")

    # 2. Load Metadata
    logger.info("Loading metadata CSVs...")
    # We merge train and val metadata to perform our own 5-fold CV on the full dataset
    df_train_part = pd.read_csv(config.train_metadata_path)
    df_val_part = pd.read_csv(config.val_metadata_path)
    train_df = pd.concat([df_train_part, df_val_part], ignore_index=True)

    test_df = pd.read_csv(config.test_metadata_path)

    # 3. Debug Subsampling
    if config.debug:
        logger.info("Debug mode enabled: Subsampling data.")
        train_df = train_df.sample(
            n=min(100, len(train_df)), random_state=config.seed
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(50, len(test_df)), random_state=config.seed
        ).reset_index(drop=True)

    # 4. Feature Engineering
    logger.info("Generating meta-features...")
    train_df = get_meta_features(train_df)
    test_df = get_meta_features(test_df)

    # 5. Create Folds
    logger.info(f"Creating {config.n_folds} stratified folds...")
    train_df = make_folds(train_df, config)

    # 6. Cache Data
    logger.info(f"Saving processed data to cache: {config.cache_dir}")
    os.makedirs(config.cache_dir, exist_ok=True)
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df
