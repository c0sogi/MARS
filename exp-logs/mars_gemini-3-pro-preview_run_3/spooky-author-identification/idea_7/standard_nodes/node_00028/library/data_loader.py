import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import string
from library.config import Config

# --- Helper Functions for Caching Sparse Matrices ---


def save_sparse_npy(matrix, base_filename):
    """
    Saves a scipy.sparse.csr_matrix to a set of .npy files.
    """
    matrix = sparse.csr_matrix(matrix)
    np.save(f"{base_filename}_data.npy", matrix.data)
    np.save(f"{base_filename}_indices.npy", matrix.indices)
    np.save(f"{base_filename}_indptr.npy", matrix.indptr)
    np.save(f"{base_filename}_shape.npy", matrix.shape)


def load_sparse_npy(base_filename):
    """
    Loads a scipy.sparse.csr_matrix from a set of .npy files.
    """
    data = np.load(f"{base_filename}_data.npy")
    indices = np.load(f"{base_filename}_indices.npy")
    indptr = np.load(f"{base_filename}_indptr.npy")
    shape = np.load(f"{base_filename}_shape.npy")
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def check_sparse_cache_exists(base_filename):
    """Checks if all components of a sparse matrix exist."""
    suffixes = ["_data.npy", "_indices.npy", "_indptr.npy", "_shape.npy"]
    return all(os.path.exists(f"{base_filename}{s}") for s in suffixes)


# --- Data Loading & Feature Extraction ---


def load_raw_data():
    """
    Loads the raw train, validation, and test data from the metadata directory.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Handle NaN in text if any (though unlikely based on analysis)
    train_df["text"] = train_df["text"].fillna("")
    val_df["text"] = val_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    return train_df, val_df, test_df


def compute_stylometric_features_raw(texts):
    """
    Computes dense stylometric features for a list of texts.
    Features: Log Char Count, Punctuation Density, Type-Token Ratio.
    """
    features = []
    punctuation_set = set(string.punctuation)

    for text in texts:
        text_len = len(text)
        # 1. Log Character Count
        log_char_count = np.log1p(text_len)

        # 2. Punctuation Density
        punct_count = sum(1 for char in text if char in punctuation_set)
        punct_density = punct_count / text_len if text_len > 0 else 0.0

        # 3. Type-Token Ratio
        tokens = text.split()
        num_tokens = len(tokens)
        num_unique_tokens = len(set(tokens))
        ttr = num_unique_tokens / num_tokens if num_tokens > 0 else 0.0

        features.append([log_char_count, punct_density, ttr])

    return np.array(features, dtype=np.float32)


def get_stylometric_features(df, dataset_name, load_cached_data=True):
    """
    Orchestrates the computation and caching of stylometric features.

    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column.
        dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The dense feature matrix.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"stylometric_{dataset_name}.npy")

    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached stylometric features for {dataset_name}...")
        return np.load(cache_path)

    # print(f"Computing stylometric features for {dataset_name}...")
    features = compute_stylometric_features_raw(df["text"].tolist())

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, features)

    return features


def get_tfidf_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates and caches TF-IDF features (Word + Char) for all splits.
    Vectorizers are fitted on the Training set only.

    Args:
        train_df, val_df, test_df: DataFrames.
        load_cached_data (bool): Whether to use cache.

    Returns:
        tuple: (train_sparse, val_sparse, test_sparse)
    """
    cache_base_train = os.path.join(Config.CACHE_DIR, "tfidf_train")
    cache_base_val = os.path.join(Config.CACHE_DIR, "tfidf_val")
    cache_base_test = os.path.join(Config.CACHE_DIR, "tfidf_test")

    if (
        load_cached_data
        and check_sparse_cache_exists(cache_base_train)
        and check_sparse_cache_exists(cache_base_val)
        and check_sparse_cache_exists(cache_base_test)
    ):
        # print("Loading cached TF-IDF features...")
        train_feats = load_sparse_npy(cache_base_train)
        val_feats = load_sparse_npy(cache_base_val)
        test_feats = load_sparse_npy(cache_base_test)
        return train_feats, val_feats, test_feats

    # print("Computing TF-IDF features...")
    train_text = train_df["text"].tolist()
    val_text = val_df["text"].tolist()
    test_text = test_df["text"].tolist()

    # 1. Word TF-IDF
    word_vectorizer = TfidfVectorizer(**Config.TFIDF_WORD_PARAMS)
    word_vectorizer.fit(train_text)
    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    # 2. Char TF-IDF
    char_vectorizer = TfidfVectorizer(**Config.TFIDF_CHAR_PARAMS)
    char_vectorizer.fit(train_text)
    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # 3. Concatenate
    train_feats = sparse.hstack([train_word, train_char]).tocsr()
    val_feats = sparse.hstack([val_word, val_char]).tocsr()
    test_feats = sparse.hstack([test_word, test_char]).tocsr()

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    save_sparse_npy(train_feats, cache_base_train)
    save_sparse_npy(val_feats, cache_base_val)
    save_sparse_npy(test_feats, cache_base_test)

    return train_feats, val_feats, test_feats


# --- PyTorch Datasets ---


class AuthorDataset(Dataset):
    """
    Dataset for supervised training and inference.
    Handles tokenization and label encoding.
    """

    def __init__(self, df, tokenizer, max_length=Config.MAX_LENGTH, is_test=False):
        self.texts = df["text"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        if not self.is_test:
            self.labels = df["author"].map(Config.LABEL2ID).tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (DAPT).
    Returns tokenized inputs; masking is typically handled by the DataCollator.
    """

    def __init__(self, texts, tokenizer, max_length=Config.MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # For MLM, we just need to return the tokenized text.
        # The DataCollatorForLanguageModeling will handle the masking.
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            # Labels are not needed here if using DataCollator,
            # but usually DataCollator expects 'input_ids' and creates 'labels' from them.
        }


def get_mlm_dataset(tokenizer, load_cached_data=True):
    """
    Helper to create the combined Train + Test dataset for MLM.
    """
    train_df, _, test_df = load_raw_data()

    # Concatenate all text
    all_texts = train_df["text"].tolist() + test_df["text"].tolist()

    return MLMDataset(all_texts, tokenizer)
