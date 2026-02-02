import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed


def load_raw_data(debug=Config.DEBUG, sample_size=Config.DEBUG_SAMPLE_SIZE):
    """
    Loads raw data from the metadata directory.

    Args:
        debug (bool): If True, subsamples the data for debugging.
        sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if debug:
        train_df = train_df.head(sample_size).copy()
        val_df = val_df.head(sample_size).copy()
        test_df = test_df.head(sample_size).copy()
        print(
            f"DEBUG MODE: Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples."
        )

    return train_df, val_df, test_df


def save_sparse_csr(path_prefix, matrix):
    """
    Saves a scipy.sparse.csr_matrix to .npy files (data, indices, indptr, shape).
    Avoids using pickle.
    """
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    np.save(f"{path_prefix}_data.npy", matrix.data)
    np.save(f"{path_prefix}_indices.npy", matrix.indices)
    np.save(f"{path_prefix}_indptr.npy", matrix.indptr)
    np.save(f"{path_prefix}_shape.npy", np.array(matrix.shape))


def load_sparse_csr(path_prefix):
    """
    Loads a scipy.sparse.csr_matrix from .npy files.
    """
    try:
        data = np.load(f"{path_prefix}_data.npy")
        indices = np.load(f"{path_prefix}_indices.npy")
        indptr = np.load(f"{path_prefix}_indptr.npy")
        shape = np.load(f"{path_prefix}_shape.npy")
        return sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))
    except FileNotFoundError:
        return None


def get_tfidf_vectors(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads TF-IDF vectors for the statistical branch.
    Combines Word N-grams and Character N-grams.

    Args:
        train_df, val_df, test_df: Pandas DataFrames containing 'text'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, X_val, X_test) as scipy sparse matrices.
    """
    cache_prefix_train = os.path.join(Config.CACHE_DIR, "X_train_tfidf")
    cache_prefix_val = os.path.join(Config.CACHE_DIR, "X_val_tfidf")
    cache_prefix_test = os.path.join(Config.CACHE_DIR, "X_test_tfidf")

    # 1. Try to load from cache
    if load_cached_data:
        X_train = load_sparse_csr(cache_prefix_train)
        X_val = load_sparse_csr(cache_prefix_val)
        X_test = load_sparse_csr(cache_prefix_test)

        if X_train is not None and X_val is not None and X_test is not None:
            # Verify dimensions match current data (Cite debug_lesson_1)
            if (
                X_train.shape[0] == len(train_df)
                and X_val.shape[0] == len(val_df)
                and X_test.shape[0] == len(test_df)
            ):
                print("Loaded TF-IDF vectors from cache.")
                return X_train, X_val, X_test
            else:
                print("Cached TF-IDF vectors dimension mismatch. Recomputing...")

    print("Computing TF-IDF vectors...")

    # 2. Compute from scratch
    # Word TF-IDF
    word_vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)
    word_vectorizer.fit(train_df["text"])

    train_word = word_vectorizer.transform(train_df["text"])
    val_word = word_vectorizer.transform(val_df["text"])
    test_word = word_vectorizer.transform(test_df["text"])

    # Char TF-IDF
    char_vectorizer = TfidfVectorizer(**Config.CHAR_TFIDF_PARAMS)
    char_vectorizer.fit(train_df["text"])

    train_char = char_vectorizer.transform(train_df["text"])
    val_char = char_vectorizer.transform(val_df["text"])
    test_char = char_vectorizer.transform(test_df["text"])

    # Combine
    X_train = sparse.hstack([train_word, train_char]).tocsr()
    X_val = sparse.hstack([val_word, val_char]).tocsr()
    X_test = sparse.hstack([test_word, test_char]).tocsr()

    # 3. Save to cache
    save_sparse_csr(cache_prefix_train, X_train)
    save_sparse_csr(cache_prefix_val, X_val)
    save_sparse_csr(cache_prefix_test, X_test)
    print(f"Saved TF-IDF vectors to {Config.CACHE_DIR}")

    return X_train, X_val, X_test


class AuthorshipDataset(Dataset):
    """
    PyTorch Dataset for the Transformer branch.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.texts = df["text"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            # Map string labels to integers
            self.labels = df["author"].map(Config.LABEL_MAP).values

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


def get_dataloaders(
    train_df,
    val_df,
    test_df,
    tokenizer_name=Config.MODEL_NAME,
    batch_size=Config.TRAIN_BATCH_SIZE,
    val_batch_size=Config.VALID_BATCH_SIZE,
    max_len=Config.MAX_LEN,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for the Transformer branch.

    Args:
        train_df, val_df, test_df: Pandas DataFrames.
        tokenizer_name (str): HuggingFace tokenizer name.
        batch_size (int): Training batch size.
        val_batch_size (int): Validation/Test batch size.
        max_len (int): Maximum sequence length.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = AuthorshipDataset(train_df, tokenizer, max_len, is_test=False)
    val_dataset = AuthorshipDataset(val_df, tokenizer, max_len, is_test=False)
    test_dataset = AuthorshipDataset(test_df, tokenizer, max_len, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
