import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import PreTrainedTokenizerBase

from library.config import Config
from library.utils import CacheManager, get_logger

logger = get_logger("DataModule")


def get_meta_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts scalar meta-features from the 'full_text' column.
    Matches the features defined in Config.meta_features:
    ['char_count', 'word_count', 'sentence_count', 'unique_word_ratio', 'paragraph_count']
    """
    texts = df["full_text"].astype(str).values

    # Pre-allocate array
    features = np.zeros((len(texts), 5), dtype=np.float32)

    for i, text in enumerate(texts):
        # 1. Char Count
        char_count = len(text)

        # 2. Word Count (simple split)
        words = text.split()
        word_count = len(words)

        # 3. Sentence Count (approximation using punctuation)
        # We look for sentence terminators.
        sentence_count = len(re.findall(r"[.!?]+", text))
        if sentence_count == 0:
            sentence_count = 1  # Fallback

        # 4. Unique Word Ratio
        if word_count > 0:
            unique_word_ratio = len(set(words)) / word_count
        else:
            unique_word_ratio = 0.0

        # 5. Paragraph Count
        # Assuming paragraphs are separated by newlines
        paragraph_count = text.count("\n") + 1

        features[i] = [
            char_count,
            word_count,
            sentence_count,
            unique_word_ratio,
            paragraph_count,
        ]

    return features


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Returns input_ids, attention_mask, meta_features, and labels (if available).
    """

    def __init__(
        self, input_ids, attention_mask, meta_features, scores=None, essay_ids=None
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.meta_features = meta_features
        self.scores = scores
        self.essay_ids = essay_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "meta_features": torch.tensor(self.meta_features[idx], dtype=torch.float32),
        }

        if self.scores is not None:
            item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float32)

        if self.essay_ids is not None:
            item["essay_id"] = self.essay_ids[idx]

        return item


def load_processed_data(
    tokenizer: PreTrainedTokenizerBase,
    mode: str = "train",
    load_cached_data: bool = True,
):
    """
    Loads data, extracts features, tokenizes, and caches the result.

    Args:
        tokenizer: Transformer tokenizer.
        mode: 'train' (loads train+val metadata) or 'test' (loads test metadata).
        load_cached_data: If True, attempts to load from cache first.

    Returns:
        EssayDataset object.
    """
    cache_manager = CacheManager(Config.cache_dir)

    # Configuration dict for hashing
    config_dict = {
        "mode": mode,
        "model_name": Config.model_name,
        "max_length": Config.max_length,
        "debug": Config.debug,
        "meta_features": Config.meta_features,
    }

    # Define cache keys
    keys = ["input_ids", "attention_mask", "meta_features", "essay_ids"]
    if mode == "train":
        keys.append("scores")

    # 1. Try Loading from Cache
    if load_cached_data:
        data = {}
        all_exist = True
        for key in keys:
            loaded = cache_manager.load(key, config_dict=config_dict, ext="npy")
            if loaded is None:
                all_exist = False
                break
            data[key] = loaded

        if all_exist:
            logger.info(f"Loaded {mode} data from cache.")
            scores = data.get("scores")
            return EssayDataset(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                meta_features=data["meta_features"],
                scores=scores,
                essay_ids=data["essay_ids"],
            )

    # 2. Process from Scratch
    logger.info(f"Processing {mode} data from scratch...")

    # Load DataFrames
    if mode == "train":
        # Combine train and val metadata for full cross-validation
        df_train = pd.read_csv(Config.train_metadata_path)
        df_val = pd.read_csv(Config.val_metadata_path)
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df = pd.read_csv(Config.test_metadata_path)

    # Handle Debug Mode
    if Config.debug:
        logger.info("Debug mode enabled: sampling 100 rows.")
        df = df.iloc[:100].reset_index(drop=True)

    # Extract Meta Features
    logger.info("Extracting meta-features...")
    meta_features = get_meta_features(df)

    # Tokenization
    logger.info("Tokenizing text...")
    texts = df["full_text"].astype(str).tolist()

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=Config.max_length,
        return_tensors="np",
        return_attention_mask=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    essay_ids = df["essay_id"].values

    # Handle Scores
    scores = None
    if mode == "train":
        scores = df["score"].values.astype(np.float32)

    # 3. Save to Cache
    logger.info("Saving processed data to cache...")
    cache_manager.save(input_ids, "input_ids", config_dict=config_dict, ext="npy")
    cache_manager.save(
        attention_mask, "attention_mask", config_dict=config_dict, ext="npy"
    )
    cache_manager.save(
        meta_features, "meta_features", config_dict=config_dict, ext="npy"
    )
    cache_manager.save(essay_ids, "essay_ids", config_dict=config_dict, ext="npy")

    if scores is not None:
        cache_manager.save(scores, "scores", config_dict=config_dict, ext="npy")

    return EssayDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        meta_features=meta_features,
        scores=scores,
        essay_ids=essay_ids,
    )


def get_folds(dataset: EssayDataset, n_folds: int = 5, seed: int = 42):
    """
    Generates Stratified K-Fold indices based on scores.

    Args:
        dataset: The training EssayDataset containing scores.
        n_folds: Number of folds.
        seed: Random seed.

    Returns:
        Generator yielding (train_indices, val_indices).
    """
    if dataset.scores is None:
        raise ValueError("Dataset must have scores to perform stratified splitting.")

    # We need integer scores for stratification
    # Assuming scores are floats in the dataset, we cast to int for stratification labels
    y = dataset.scores.astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Dummy X (zeros) since we only need indices based on y
    X = np.zeros(len(y))

    return skf.split(X, y)
