import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from transformers import AutoTokenizer

from library.configuration import Config
from library.utilities import seed_everything


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for Author Identification.
    Handles tokenization using Hugging Face transformers.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'text' and optionally 'author'.
            tokenizer: Hugging Face tokenizer instance.
            max_len (int): Maximum sequence length.
            is_test (bool): If True, does not look for 'author' column.
        """
        self.texts = df["text"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            self.labels = df["author"].map(Config.LABEL2ID).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def get_stratified_folds(df, num_folds=5, seed=42):
    """
    Assigns fold numbers to the dataframe using Stratified K-Fold.

    Args:
        df (pd.DataFrame): DataFrame containing 'author' column.
        num_folds (int): Number of folds.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'fold' column.
    """
    seed_everything(seed)

    df = df.copy()
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # We stratify based on the 'author' label
    for fold, (_, val_idx) in enumerate(skf.split(df, df["author"])):
        df.loc[val_idx, "fold"] = fold

    return df


def get_tfidf_features(train_df, test_df, load_cached_data=True):
    """
    Generates or loads sparse TF-IDF features for training and test sets.
    Combines Word N-grams (1-3) and Character N-grams (3-5).

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_sparse, X_test_sparse)
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.WORKING_DIR, "tfidf_train.npz")
    test_cache_path = os.path.join(Config.WORKING_DIR, "tfidf_test.npz")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        try:
            X_train = sparse.load_npz(train_cache_path)
            X_test = sparse.load_npz(test_cache_path)

            # Simple validation to ensure cache matches current data shape
            if X_train.shape[0] == len(train_df) and X_test.shape[0] == len(test_df):
                return X_train, X_test
        except Exception:
            pass  # Fallback to computation

    # 2. Compute features from scratch
    # Word-level TF-IDF
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
        max_features=Config.TFIDF_MAX_FEATURES_WORD,
        sublinear_tf=True,
    )

    # Char-level TF-IDF
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE,
        max_features=Config.TFIDF_MAX_FEATURES_CHAR,
        sublinear_tf=True,
    )

    # Fit on training data
    # Note: We fit strictly on training data to avoid leakage,
    # though fitting on full corpus for vocabulary is sometimes done in competitions.
    # We stick to rigorous fit-on-train.
    all_train_text = train_df["text"].fillna("").astype(str)
    all_test_text = test_df["text"].fillna("").astype(str)

    # Fit and Transform Words
    X_train_word = word_vectorizer.fit_transform(all_train_text)
    X_test_word = word_vectorizer.transform(all_test_text)

    # Fit and Transform Chars
    X_train_char = char_vectorizer.fit_transform(all_train_text)
    X_test_char = char_vectorizer.transform(all_test_text)

    # Stack features
    X_train = sparse.hstack([X_train_word, X_train_char]).tocsr()
    X_test = sparse.hstack([X_test_word, X_test_char]).tocsr()

    # 3. Save to cache
    try:
        sparse.save_npz(train_cache_path, X_train)
        sparse.save_npz(test_cache_path, X_test)
    except Exception as e:
        print(f"Warning: Failed to save TF-IDF cache. Error: {e}")

    return X_train, X_test
