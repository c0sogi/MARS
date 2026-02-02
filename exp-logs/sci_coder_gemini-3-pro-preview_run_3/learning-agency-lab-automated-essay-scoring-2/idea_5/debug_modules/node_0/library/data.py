import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import get_hash


def get_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes explicit meta-features from the 'full_text' column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'full_text' column.

    Returns:
        pd.DataFrame: DataFrame containing only the computed meta-features.
    """
    # Ensure text is string
    texts = df["full_text"].astype(str).fillna("")

    features = pd.DataFrame(index=df.index)

    # Character Count
    features["char_count"] = texts.apply(len)

    # Word Count
    # Using simple split by whitespace
    features["word_count"] = texts.apply(lambda x: len(x.split()))

    # Sentence Count
    # Approximation using punctuation counts
    features["sentence_count"] = texts.apply(
        lambda x: x.count(".") + x.count("?") + x.count("!")
    )

    # Unique Word Ratio
    def unique_ratio(text):
        words = text.split()
        if len(words) == 0:
            return 0.0
        return len(set(words)) / len(words)

    features["unique_word_ratio"] = texts.apply(unique_ratio)

    return features


def process_data(config: Config, load_cached_data: bool = True):
    """
    Loads raw data, combines train/val splits, generates stratified folds,
    computes meta-features, and handles caching.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, test_df)
            - train_df: DataFrame with columns [essay_id, full_text, score, fold, ...meta_features]
            - test_df: DataFrame with columns [essay_id, full_text, ...meta_features]
    """
    # Generate a hash based on relevant config parameters to ensure cache validity
    # We include seed, n_folds, debug status, and meta_features list in the hash
    config_dict = {
        "seed": config.seed,
        "n_folds": config.n_folds,
        "debug": config.debug,
        "debug_subset_size": config.debug_subset_size,
        "meta_features": config.meta_features,
    }
    config_hash = get_hash(config_dict)

    # Define cache paths
    os.makedirs(config.cache_dir, exist_ok=True)
    train_cache_path = os.path.join(
        config.cache_dir, f"train_processed_{config_hash}.parquet"
    )
    test_cache_path = os.path.join(
        config.cache_dir, f"test_processed_{config_hash}.parquet"
    )

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            print(f"Loading processed data from cache: {config.cache_dir}")
            try:
                train_df = pd.read_parquet(train_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing data...")
        else:
            print("Cache not found. Processing data from scratch...")
    else:
        print("Forcing data processing (ignoring cache)...")

    # 2. Load Raw Data
    # We combine the provided train and val metadata to perform our own 5-fold CV
    df_train_meta = pd.read_csv(config.train_metadata_path)
    df_val_meta = pd.read_csv(config.val_metadata_path)
    train_df = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(drop=True)

    test_df = pd.read_csv(config.test_metadata_path)

    # 3. Handle Debug Mode
    if config.debug:
        print(f"Debug mode enabled. Sampling {config.debug_subset_size} rows.")
        train_df = train_df.sample(
            n=min(len(train_df), config.debug_subset_size), random_state=config.seed
        ).reset_index(drop=True)
        # For test, we also sample if it's too large, or just keep it
        test_df = test_df.sample(
            n=min(len(test_df), config.debug_subset_size), random_state=config.seed
        ).reset_index(drop=True)

    # 4. Create Stratified Folds
    # We use StratifiedKFold on the 'score' column
    train_df["fold"] = -1
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["score"])):
        train_df.loc[val_idx, "fold"] = fold

    train_df["fold"] = train_df["fold"].astype(int)

    # 5. Compute Meta-Features
    print("Computing meta-features...")
    train_meta = get_meta_features(train_df)
    test_meta = get_meta_features(test_df)

    # Concatenate meta-features to the main DataFrames
    train_df = pd.concat([train_df, train_meta], axis=1)
    test_df = pd.concat([test_df, test_meta], axis=1)

    # 6. Save to Cache
    print(f"Saving processed data to cache: {config.cache_dir}")
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Tokenizes text on-the-fly and provides input tensors and targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        tokenizer: AutoTokenizer,
        is_test: bool = False,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'full_text' and 'essay_id'.
                               Must contain 'score' if is_test is False.
            config (Config): Configuration object.
            tokenizer (AutoTokenizer): Pre-trained tokenizer.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = df
        self.config = config
        self.tokenizer = tokenizer
        self.is_test = is_test

        # Pre-extract data to lists for faster access in __getitem__
        self.texts = df["full_text"].astype(str).tolist()
        self.essay_ids = df["essay_id"].tolist()

        if not self.is_test:
            self.scores = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenization
        # We use padding='max_length' to ensure consistent tensor shapes within batches
        # Truncation handles texts longer than max_length
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        # Squeeze to remove the batch dimension added by return_tensors='pt'
        # Shape becomes (max_length,) instead of (1, max_length)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_ids": self.essay_ids[idx],
        }

        if not self.is_test:
            # Targets for regression (MSE loss)
            sample["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return sample
