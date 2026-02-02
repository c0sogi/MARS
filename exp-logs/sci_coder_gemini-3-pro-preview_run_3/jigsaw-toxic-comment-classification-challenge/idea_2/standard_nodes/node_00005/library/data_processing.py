import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from library.utils import seed_everything

# Configuration Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_2"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_data_from_metadata():
    """
    Loads training, validation, and test data using metadata to map to source text.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # Load Source Data
    # We read the full CSVs once to extract text by index.
    # This is memory efficient enough for this dataset size.
    orig_train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    orig_test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # Helper to merge text
    def attach_text(meta_df, source_df):
        # We use the source_row_index to fetch the text
        # Ensure indices align. The source_row_index corresponds to the implicit index of the source csv.
        texts = source_df.iloc[meta_df["source_row_index"].values][
            "comment_text"
        ].values

        # Create a copy to avoid SettingWithCopy warnings
        out_df = meta_df.copy()
        out_df["comment_text"] = texts

        # Fill NaNs in text
        out_df["comment_text"] = out_df["comment_text"].fillna("fillna")
        return out_df

    train_df = attach_text(train_meta, orig_train)
    val_df = attach_text(val_meta, orig_train)
    test_df = attach_text(test_meta, orig_test)

    return train_df, val_df, test_df


def get_tfidf_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads TF-IDF features for NBSVM.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (train_features, val_features, test_features) as scipy sparse matrices.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_path = os.path.join(CACHE_DIR, "tfidf_train.npz")
    val_path = os.path.join(CACHE_DIR, "tfidf_val.npz")
    test_path = os.path.join(CACHE_DIR, "tfidf_test.npz")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    ):
        print("Loading TF-IDF features from cache...")
        train_features = sparse.load_npz(train_path)
        val_features = sparse.load_npz(val_path)
        test_features = sparse.load_npz(test_path)
        return train_features, val_features, test_features

    print("Computing TF-IDF features...")

    # Concatenate train and val for fitting to get a better vocabulary representation
    # (Standard practice in this competition is often to fit on Train+Test,
    # but strictly we fit on Train+Val to avoid leakage, or just Train.
    # Here we fit on Train+Val).
    train_text = train_df["comment_text"]
    val_text = val_df["comment_text"]
    test_text = test_df["comment_text"]

    all_text = pd.concat([train_text, val_text])

    # Word Vectorizer
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )

    # Char Vectorizer
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="char",
        stop_words="english",
        ngram_range=(2, 6),
        max_features=30000,
    )

    # Fit and Transform
    # Note: Fitting on Train+Val
    print("Fitting Word Vectorizer...")
    word_vectorizer.fit(all_text)
    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    print("Fitting Char Vectorizer...")
    char_vectorizer.fit(all_text)
    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # Stack features
    train_features = sparse.hstack([train_word, train_char])
    val_features = sparse.hstack([val_word, val_char])
    test_features = sparse.hstack([test_word, test_char])

    # Cache results
    print("Saving TF-IDF features to cache...")
    sparse.save_npz(train_path, train_features)
    sparse.save_npz(val_path, val_features)
    sparse.save_npz(test_path, test_features)

    return train_features, val_features, test_features


class ToxicDataset(Dataset):
    """
    PyTorch Dataset for Toxic Comment Classification (RoBERTa).
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.text = df["comment_text"].values

        if not self.is_test:
            self.targets = df[LABEL_COLS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())  # normalize whitespace

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,  # RoBERTa doesn't use token_type_ids usually
            return_attention_mask=True,
            return_tensors="pt",
        )

        ids = inputs["input_ids"].flatten()
        mask = inputs["attention_mask"].flatten()

        item = {"ids": ids, "mask": mask}

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[index], dtype=torch.float)

        return item


def make_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=16, max_len=128):
    """
    Creates PyTorch DataLoaders.

    Args:
        train_df, val_df, test_df: DataFrames containing data.
        tokenizer: Transformer tokenizer.
        batch_size: Batch size.
        max_len: Maximum sequence length.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = ToxicDataset(train_df, tokenizer, max_len, is_test=False)
    val_dataset = ToxicDataset(val_df, tokenizer, max_len, is_test=False)
    test_dataset = ToxicDataset(test_df, tokenizer, max_len, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
