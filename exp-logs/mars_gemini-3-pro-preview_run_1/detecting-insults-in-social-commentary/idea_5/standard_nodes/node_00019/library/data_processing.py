import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from library.config import Config


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Returns input_ids, attention_mask, svd_features, and labels (if available).
    """

    def __init__(self, texts, svd_features, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.svd_features = svd_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        svd_vec = self.svd_features[idx]

        # Tokenize
        encoding = self.tokenizer(
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
            "svd_features": torch.tensor(svd_vec, dtype=torch.float),
        }

        if self.labels is not None:
            # Use float for BCEWithLogitsLoss
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def _get_structural_features(train_texts, val_texts, test_texts, load_cached_data=True):
    """
    Generates or loads TF-IDF + SVD features.
    Fits ONLY on train_texts to avoid leakage.
    """
    # Define cache paths
    os.makedirs(Config.working_dir, exist_ok=True)
    cache_train_path = os.path.join(Config.working_dir, "meta_train_svd.npy")
    cache_val_path = os.path.join(Config.working_dir, "meta_val_svd.npy")
    cache_test_path = os.path.join(Config.working_dir, "meta_test_svd.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print("Loading cached SVD features...")
            return (
                np.load(cache_train_path),
                np.load(cache_val_path),
                np.load(cache_test_path),
            )

    # 2. Compute from scratch
    print("Computing Structural Features (TF-IDF + SVD) from scratch...")

    # Initialize Vectorizers
    print(
        f"  Fitting TF-IDF (Word: {Config.tfidf_word_ngram_range}, Char: {Config.tfidf_char_ngram_range})..."
    )
    word_vec = TfidfVectorizer(
        ngram_range=Config.tfidf_word_ngram_range,
        min_df=2,
        analyzer="word",
        token_pattern=r"\w{1,}",
    )
    char_vec = TfidfVectorizer(
        ngram_range=Config.tfidf_char_ngram_range, min_df=2, analyzer="char"
    )

    # Fit on Train ONLY
    word_vec.fit(train_texts)
    char_vec.fit(train_texts)

    # Transform all splits
    print("  Transforming texts to sparse matrices...")
    train_word = word_vec.transform(train_texts)
    train_char = char_vec.transform(train_texts)

    val_word = word_vec.transform(val_texts)
    val_char = char_vec.transform(val_texts)

    test_word = word_vec.transform(test_texts)
    test_char = char_vec.transform(test_texts)

    # Concatenate Word and Char features
    train_sparse = sparse.hstack([train_word, train_char])
    val_sparse = sparse.hstack([val_word, val_char])
    test_sparse = sparse.hstack([test_word, test_char])

    # Fit TruncatedSVD on Train ONLY
    print(f"  Fitting TruncatedSVD (components={Config.svd_components})...")
    svd = TruncatedSVD(n_components=Config.svd_components, random_state=Config.seed)
    svd.fit(train_sparse)

    # Transform to dense SVD features
    train_svd = svd.transform(train_sparse)
    val_svd = svd.transform(val_sparse)
    test_svd = svd.transform(test_sparse)

    # Normalize (StandardScaler) - Fit on Train ONLY
    print("  Normalizing features...")
    scaler = StandardScaler()
    train_svd = scaler.fit_transform(train_svd)
    val_svd = scaler.transform(val_svd)
    test_svd = scaler.transform(test_svd)

    # 3. Save to cache
    print(f"  Saving features to {Config.working_dir}...")
    np.save(cache_train_path, train_svd)
    np.save(cache_val_path, val_svd)
    np.save(cache_test_path, test_svd)

    return train_svd, val_svd, test_svd


def load_data(load_cached_data=True):
    """
    Main entry point to load datasets.
    Reads metadata CSVs, processes features, and returns PyTorch Datasets.
    """
    print("Loading Metadata CSVs...")
    # Load metadata
    try:
        df_train = pd.read_csv(Config.metadata_train_path)
        df_val = pd.read_csv(Config.metadata_val_path)
        # Assuming test.csv is in metadata as per generation script
        test_meta_path = os.path.join(
            os.path.dirname(Config.metadata_train_path), "test.csv"
        )
        df_test = pd.read_csv(test_meta_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Metadata files not found. Ensure metadata generation was successful. Error: {e}"
        )

    # Handle missing values in text
    df_train["Comment"] = df_train["Comment"].fillna("").astype(str)
    df_val["Comment"] = df_val["Comment"].fillna("").astype(str)
    df_test["Comment"] = df_test["Comment"].fillna("").astype(str)

    # Get Structural Features (SVD)
    train_svd, val_svd, test_svd = _get_structural_features(
        df_train["Comment"].tolist(),
        df_val["Comment"].tolist(),
        df_test["Comment"].tolist(),
        load_cached_data=load_cached_data,
    )

    print(f"SVD Feature Shape: {train_svd.shape}")

    # Initialize Tokenizer
    print(f"Initializing Tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    print("Creating PyTorch Datasets...")
    train_dataset = InsultDataset(
        texts=df_train["Comment"].values,
        svd_features=train_svd,
        labels=df_train["Insult"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
    )

    val_dataset = InsultDataset(
        texts=df_val["Comment"].values,
        svd_features=val_svd,
        labels=df_val["Insult"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
    )

    test_dataset = InsultDataset(
        texts=df_test["Comment"].values,
        svd_features=test_svd,
        labels=None,
        tokenizer=tokenizer,
        max_len=Config.max_len,
    )

    return train_dataset, val_dataset, test_dataset
