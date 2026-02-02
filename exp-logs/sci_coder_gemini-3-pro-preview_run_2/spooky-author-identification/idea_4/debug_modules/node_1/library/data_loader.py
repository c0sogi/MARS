import os
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    TFIDF_MIN_DF,
    SVD_COMPONENTS,
    WORD_NGRAM_RANGE,
    CHAR_NGRAM_RANGE,
    SEED,
    MAX_LENGTH,
)


def load_data():
    """
    Loads the train, validation, and test datasets from the metadata directory.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def get_tfidf_features(train_text, val_text, test_text, load_cached_data=True):
    """
    Generates or loads sparse TF-IDF features (Word + Char N-grams).

    Args:
        train_text (pd.Series): Training text data.
        val_text (pd.Series): Validation text data.
        test_text (pd.Series): Test text data.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (train_tfidf, val_tfidf, test_tfidf) as scipy.sparse matrices.
    """
    # Define cache paths
    cache_train = os.path.join(WORKING_DIR, "tfidf_train.npz")
    cache_val = os.path.join(WORKING_DIR, "tfidf_val.npz")
    cache_test = os.path.join(WORKING_DIR, "tfidf_test.npz")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):
        print("Loading cached TF-IDF features...")
        train_tfidf = sp.load_npz(cache_train)
        val_tfidf = sp.load_npz(cache_val)
        test_tfidf = sp.load_npz(cache_test)
        return train_tfidf, val_tfidf, test_tfidf

    print("Computing TF-IDF features...")

    # Word N-grams
    word_vectorizer = TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        ngram_range=WORD_NGRAM_RANGE,
        analyzer="word",
        token_pattern=r"\w{1,}",
        sublinear_tf=True,
    )

    # Char N-grams
    char_vectorizer = TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        ngram_range=CHAR_NGRAM_RANGE,
        analyzer="char",
        sublinear_tf=True,
    )

    # Fit on training data only to prevent leakage
    print("Fitting Word Vectorizer...")
    word_vectorizer.fit(train_text)
    print("Fitting Char Vectorizer...")
    char_vectorizer.fit(train_text)

    # Transform all splits
    print("Transforming text data...")
    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # Concatenate features
    train_tfidf = sp.hstack([train_word, train_char], format="csr")
    val_tfidf = sp.hstack([val_word, val_char], format="csr")
    test_tfidf = sp.hstack([test_word, test_char], format="csr")

    # Save to cache
    print("Saving TF-IDF features to cache...")
    sp.save_npz(cache_train, train_tfidf)
    sp.save_npz(cache_val, val_tfidf)
    sp.save_npz(cache_test, test_tfidf)

    return train_tfidf, val_tfidf, test_tfidf


def get_svd_features(train_tfidf, val_tfidf, test_tfidf, load_cached_data=True):
    """
    Generates or loads dense SVD features from sparse TF-IDF matrices.

    Args:
        train_tfidf (scipy.sparse): Training TF-IDF matrix.
        val_tfidf (scipy.sparse): Validation TF-IDF matrix.
        test_tfidf (scipy.sparse): Test TF-IDF matrix.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (train_svd, val_svd, test_svd) as numpy arrays.
    """
    cache_train = os.path.join(WORKING_DIR, "svd_train.npy")
    cache_val = os.path.join(WORKING_DIR, "svd_val.npy")
    cache_test = os.path.join(WORKING_DIR, "svd_test.npy")

    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):
        print("Loading cached SVD features...")
        train_svd = np.load(cache_train)
        val_svd = np.load(cache_val)
        test_svd = np.load(cache_test)
        return train_svd, val_svd, test_svd

    print("Computing SVD features...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)

    # Fit on training data
    print("Fitting SVD...")
    svd.fit(train_tfidf)

    # Transform all splits
    print("Transforming TF-IDF to Dense SVD...")
    train_svd = svd.transform(train_tfidf)
    val_svd = svd.transform(val_tfidf)
    test_svd = svd.transform(test_tfidf)

    # Save to cache
    print("Saving SVD features to cache...")
    np.save(cache_train, train_svd)
    np.save(cache_val, val_svd)
    np.save(cache_test, test_svd)

    return train_svd, val_svd, test_svd


class TextDataset(Dataset):
    """
    PyTorch Dataset for tokenizing text on-the-fly.
    """

    def __init__(self, texts, labels=None, tokenizer=None, max_length=MAX_LENGTH):
        """
        Args:
            texts (list or np.array): List of text strings.
            labels (list or np.array, optional): List of integer labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
            max_length (int): Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        # Squeeze to remove batch dimension added by tokenizer
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
