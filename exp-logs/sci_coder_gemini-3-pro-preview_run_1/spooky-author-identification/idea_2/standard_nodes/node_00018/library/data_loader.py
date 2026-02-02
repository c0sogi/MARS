import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from library.config import Config


def load_data(split="train"):
    """
    Loads the dataset for the specified split from the metadata directory.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if split == "train":
        path = Config.TRAIN_DATA_PATH
    elif split == "val":
        path = Config.VAL_DATA_PATH
    elif split == "test":
        path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    return pd.read_csv(path)


def _save_sparse_csr(filename, matrix):
    """
    Helper to save a CSR matrix using numpy.savez to avoid pickle.
    """
    np.savez(
        filename,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,
    )


def _load_sparse_csr(filename):
    """
    Helper to load a CSR matrix from a numpy .npz file.
    """
    loader = np.load(filename)
    return sparse.csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
    )


def get_tfidf_features(train_text, val_text, test_text, load_cached_data=True):
    """
    Generates or loads TF-IDF features for the linear model branch.
    Combines Word N-grams and Character N-grams.

    Args:
        train_text (pd.Series): Training text data.
        val_text (pd.Series): Validation text data.
        test_text (pd.Series): Test text data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, X_val, X_test) as scipy.sparse.csr_matrix.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_train = os.path.join(Config.CACHE_DIR, "tfidf_train.npz")
    cache_val = os.path.join(Config.CACHE_DIR, "tfidf_val.npz")
    cache_test = os.path.join(Config.CACHE_DIR, "tfidf_test.npz")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            try:
                X_train = _load_sparse_csr(cache_train)
                X_val = _load_sparse_csr(cache_val)
                X_test = _load_sparse_csr(cache_test)
                return X_train, X_val, X_test
            except Exception:
                # If loading fails, proceed to recompute
                pass

    # Fill NaNs just in case
    train_text = train_text.fillna("").astype(str)
    val_text = val_text.fillna("").astype(str)
    test_text = test_text.fillna("").astype(str)

    # 1. Word N-grams
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
        max_features=Config.MAX_FEATURES_WORD,
        analyzer="word",
        token_pattern=r"\w{1,}",
        sublinear_tf=True,
    )

    # 2. Character N-grams
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE,
        max_features=Config.MAX_FEATURES_CHAR,
        analyzer="char",
        sublinear_tf=True,
    )

    # Fit on Train, Transform All
    # Word
    X_train_word = word_vectorizer.fit_transform(train_text)
    X_val_word = word_vectorizer.transform(val_text)
    X_test_word = word_vectorizer.transform(test_text)

    # Char
    X_train_char = char_vectorizer.fit_transform(train_text)
    X_val_char = char_vectorizer.transform(val_text)
    X_test_char = char_vectorizer.transform(test_text)

    # Combine
    X_train = sparse.hstack([X_train_word, X_train_char], format="csr")
    X_val = sparse.hstack([X_val_word, X_val_char], format="csr")
    X_test = sparse.hstack([X_test_word, X_test_char], format="csr")

    # Save to cache
    _save_sparse_csr(cache_train, X_train)
    _save_sparse_csr(cache_val, X_val)
    _save_sparse_csr(cache_test, X_test)

    return X_train, X_val, X_test


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for the Transformer branch.
    Handles tokenization and label mapping.
    """

    def __init__(
        self, texts, labels=None, tokenizer=None, max_length=Config.MAX_LENGTH
    ):
        """
        Args:
            texts (list or pd.Series): Input text sequences.
            labels (list or pd.Series, optional): Target labels (strings).
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
            max_length (int): Maximum sequence length.
        """
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = Config.LABEL2ID

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            label_str = self.labels[idx]
            label_id = self.label2id[label_str]
            item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item
