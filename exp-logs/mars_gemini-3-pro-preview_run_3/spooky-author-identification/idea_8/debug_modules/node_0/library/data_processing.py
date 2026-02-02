import os
import string
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from library.config import PathConfig, ModelConfig, TrainConfig, FeatureConfig
from library.utils import set_seed


def load_data():
    """
    Loads the train, validation, and test datasets from the metadata directory.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(PathConfig.TRAIN_CSV)
    val_df = pd.read_csv(PathConfig.VAL_CSV)
    test_df = pd.read_csv(PathConfig.TEST_CSV)
    return train_df, val_df, test_df


def get_auxiliary_targets(texts):
    """
    Computes auxiliary stylometric targets for Multi-Task Learning.
    Targets:
        1. Log-Character-Length: log(len(text) + 1)
        2. Punctuation Density: count(punctuation) / len(text)

    Args:
        texts (list or pd.Series): List of text strings.

    Returns:
        np.ndarray: Array of shape (N, 2) containing the targets.
    """
    targets = []
    punctuation_set = set(string.punctuation)

    for text in texts:
        text_len = len(text)
        # 1. Log Character Length
        log_len = np.log(text_len + 1)

        # 2. Punctuation Density
        if text_len > 0:
            punct_count = sum(1 for char in text if char in punctuation_set)
            punct_density = punct_count / text_len
        else:
            punct_density = 0.0

        targets.append([log_len, punct_density])

    return np.array(targets, dtype=np.float32)


def _save_sparse_matrix(matrix, file_prefix):
    """
    Saves a scipy sparse matrix as separate .npy files to avoid pickle.
    """
    matrix = sparse.csr_matrix(matrix)
    np.save(f"{file_prefix}_data.npy", matrix.data)
    np.save(f"{file_prefix}_indices.npy", matrix.indices)
    np.save(f"{file_prefix}_indptr.npy", matrix.indptr)
    np.save(f"{file_prefix}_shape.npy", matrix.shape)


def _load_sparse_matrix(file_prefix):
    """
    Loads a scipy sparse matrix from separate .npy files.
    """
    data = np.load(f"{file_prefix}_data.npy")
    indices = np.load(f"{file_prefix}_indices.npy")
    indptr = np.load(f"{file_prefix}_indptr.npy")
    shape = np.load(f"{file_prefix}_shape.npy")
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def get_tfidf_features(train_text, val_text, test_text, load_cached_data=True):
    """
    Generates or loads TF-IDF features (Word + Char N-grams).

    Args:
        train_text (pd.Series): Training text.
        val_text (pd.Series): Validation text.
        test_text (pd.Series): Test text.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, X_val, X_test) as scipy sparse matrices.
    """
    # Ensure working directory exists
    os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)

    # Define file prefixes
    train_prefix = os.path.join(PathConfig.WORKING_DIR, "X_train_tfidf")
    val_prefix = os.path.join(PathConfig.WORKING_DIR, "X_val_tfidf")
    test_prefix = os.path.join(PathConfig.WORKING_DIR, "X_test_tfidf")

    # Check if all files exist
    files_exist = all(
        os.path.exists(f"{p}_{s}.npy")
        for p in [train_prefix, val_prefix, test_prefix]
        for s in ["data", "indices", "indptr", "shape"]
    )

    if load_cached_data and files_exist:
        print("Loading cached TF-IDF features...")
        X_train = _load_sparse_matrix(train_prefix)
        X_val = _load_sparse_matrix(val_prefix)
        X_test = _load_sparse_matrix(test_prefix)
        return X_train, X_val, X_test

    print("Computing TF-IDF features...")

    # Combine all text for fitting to ensure consistent vocabulary
    all_text = pd.concat([train_text, val_text, test_text], axis=0)

    # 1. Word N-grams
    word_vectorizer = TfidfVectorizer(
        ngram_range=FeatureConfig.WORD_NGRAM_RANGE,
        max_features=FeatureConfig.MAX_FEATURES_WORD,
        analyzer="word",
        token_pattern=r"\w{1,}",
    )
    word_vectorizer.fit(all_text)

    # 2. Character N-grams
    char_vectorizer = TfidfVectorizer(
        ngram_range=FeatureConfig.CHAR_NGRAM_RANGE,
        max_features=FeatureConfig.MAX_FEATURES_CHAR,
        analyzer="char",
    )
    char_vectorizer.fit(all_text)

    # Transform and Stack
    def transform_and_stack(text_series):
        w_feats = word_vectorizer.transform(text_series)
        c_feats = char_vectorizer.transform(text_series)
        return sparse.hstack([w_feats, c_feats])

    X_train = transform_and_stack(train_text)
    X_val = transform_and_stack(val_text)
    X_test = transform_and_stack(test_text)

    # Cache results
    print("Caching TF-IDF features...")
    _save_sparse_matrix(X_train, train_prefix)
    _save_sparse_matrix(X_val, val_prefix)
    _save_sparse_matrix(X_test, test_prefix)

    return X_train, X_val, X_test


class StylometricDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning with Multi-Task Learning targets.
    """

    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = (
            texts.reset_index(drop=True)
            if hasattr(texts, "reset_index")
            else list(texts)
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Handle labels
        self.labels = None
        if labels is not None:
            # Convert string labels to IDs
            label_map = ModelConfig.LABEL2ID
            self.labels = [label_map[l] for l in labels]

        # Pre-compute auxiliary targets
        self.aux_targets = get_auxiliary_targets(self.texts)

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
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "aux_targets": torch.tensor(self.aux_targets[idx], dtype=torch.float),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-training (MLM).
    Simply returns tokenized text; masking is handled by DataCollator.
    """

    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = (
            texts.reset_index(drop=True)
            if hasattr(texts, "reset_index")
            else list(texts)
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # For MLM, we just return the inputs.
        # The DataCollatorForLanguageModeling will handle masking.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
