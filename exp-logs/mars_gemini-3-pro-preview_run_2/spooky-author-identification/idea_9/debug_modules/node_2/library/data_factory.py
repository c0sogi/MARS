import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.seed)


class TextDataset(Dataset):
    """
    PyTorch Dataset for Author Identification Task.
    Handles tokenization of text sequences using a pre-trained Transformer tokenizer.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'text' and optionally 'author' columns.
            tokenizer (PreTrainedTokenizer): Transformer tokenizer.
            max_len (int): Maximum sequence length.
            is_test (bool): If True, does not look for 'author' column.
        """
        self.texts = df["text"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            # Map string labels to integers using Config
            self.labels = df["author"].map(Config.label2id).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
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


def load_data():
    """
    Loads train, validation, and test datasets from metadata CSVs.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    # Debug mode: sample data if Config.debug is True
    if Config.debug:
        train_df = train_df.sample(
            n=min(len(train_df), 100), random_state=Config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), 50), random_state=Config.seed
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), 50), random_state=Config.seed
        ).reset_index(drop=True)

    return train_df, val_df, test_df


def get_classical_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads classical features (TF-IDF and SVD).

    Features:
    1. Sparse: Word N-grams + Char N-grams (for Linear Models)
    2. Dense: Truncated SVD of the Sparse matrix (for XGBoost)

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing sparse and dense features for all splits.
              Keys: 'train_sparse', 'val_sparse', 'test_sparse',
                    'train_dense', 'val_dense', 'test_dense'
    """
    # Define cache file paths
    cache_dir = Config.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_sparse": os.path.join(cache_dir, "train_features_sparse.npz"),
        "val_sparse": os.path.join(cache_dir, "val_features_sparse.npz"),
        "test_sparse": os.path.join(cache_dir, "test_features_sparse.npz"),
        "train_dense": os.path.join(cache_dir, "train_features_dense.npy"),
        "val_dense": os.path.join(cache_dir, "val_features_dense.npy"),
        "test_dense": os.path.join(cache_dir, "test_features_dense.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_exists:
        print("Loading classical features from cache...")
        data = {}
        data["train_sparse"] = sparse.load_npz(files["train_sparse"])
        data["val_sparse"] = sparse.load_npz(files["val_sparse"])
        data["test_sparse"] = sparse.load_npz(files["test_sparse"])

        data["train_dense"] = np.load(files["train_dense"])
        data["val_dense"] = np.load(files["val_dense"])
        data["test_dense"] = np.load(files["test_dense"])
        return data

    print("Computing classical features from scratch...")

    # Extract text
    train_text = train_df["text"].fillna("").astype(str)
    val_text = val_df["text"].fillna("").astype(str)
    test_text = test_df["text"].fillna("").astype(str)

    # 1. Word N-grams
    print("Vectorizing Word N-grams...")
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.tfidf_word_ngram_range,
        min_df=Config.tfidf_min_df,
        stop_words="english",
        sublinear_tf=True,
    )
    train_word = word_vectorizer.fit_transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    # 2. Char N-grams
    print("Vectorizing Char N-grams...")
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.tfidf_char_ngram_range,
        min_df=Config.tfidf_min_df,
        analyzer="char_wb",
        sublinear_tf=True,
    )
    train_char = char_vectorizer.fit_transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # Combine for Sparse Features
    print("Stacking Sparse Features...")
    train_sparse = sparse.hstack([train_word, train_char])
    val_sparse = sparse.hstack([val_word, val_char])
    test_sparse = sparse.hstack([test_word, test_char])

    # 3. Truncated SVD for Dense Features
    print(f"Fitting Truncated SVD (n_components={Config.svd_n_components})...")
    svd = TruncatedSVD(n_components=Config.svd_n_components, random_state=Config.seed)
    train_dense = svd.fit_transform(train_sparse)
    val_dense = svd.transform(val_sparse)
    test_dense = svd.transform(test_sparse)

    # Save to cache
    print("Saving features to cache...")
    sparse.save_npz(files["train_sparse"], train_sparse)
    sparse.save_npz(files["val_sparse"], val_sparse)
    sparse.save_npz(files["test_sparse"], test_sparse)

    np.save(files["train_dense"], train_dense)
    np.save(files["val_dense"], val_dense)
    np.save(files["test_dense"], test_dense)

    return {
        "train_sparse": train_sparse,
        "val_sparse": val_sparse,
        "test_sparse": test_sparse,
        "train_dense": train_dense,
        "val_dense": val_dense,
        "test_dense": test_dense,
    }
