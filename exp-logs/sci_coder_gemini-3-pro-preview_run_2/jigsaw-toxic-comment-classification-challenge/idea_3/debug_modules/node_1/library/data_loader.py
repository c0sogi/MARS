import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from library.config import Config


class ToxicDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Prediction.
    Handles tokenization and input formatting for Transformer models.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.texts = df["comment_text"].values

        if not self.is_test:
            self.targets = df[Config.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.texts[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        out = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

        if not self.is_test:
            out["targets"] = torch.tensor(self.targets[index], dtype=torch.float)

        return out


def get_dataloaders(
    tokenizer,
    train_batch_size=Config.train_batch_size,
    valid_batch_size=Config.valid_batch_size,
):
    """
    Creates Training and Validation DataLoaders.

    Args:
        tokenizer: Transformer tokenizer instance.
        train_batch_size (int): Batch size for training.
        valid_batch_size (int): Batch size for validation.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load data from metadata
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)

    # Handle Debug Mode
    if Config.debug:
        print(f"Debug Mode: Sampling {min(1000, len(train_df))} rows for training.")
        train_df = train_df.sample(
            n=min(1000, len(train_df)), random_state=Config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(500, len(val_df)), random_state=Config.seed
        ).reset_index(drop=True)

    train_dataset = ToxicDataset(train_df, tokenizer, Config.max_len, is_test=False)
    val_dataset = ToxicDataset(val_df, tokenizer, Config.max_len, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(tokenizer, test_batch_size=Config.valid_batch_size):
    """
    Creates Test DataLoader.

    Args:
        tokenizer: Transformer tokenizer instance.
        test_batch_size (int): Batch size for inference.

    Returns:
        tuple: (test_loader, test_ids)
    """
    test_df = pd.read_csv(Config.test_path)

    if Config.debug:
        print(f"Debug Mode: Sampling {min(1000, len(test_df))} rows for testing.")
        test_df = test_df.iloc[: min(1000, len(test_df))]

    test_dataset = ToxicDataset(test_df, tokenizer, Config.max_len, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_df["id"].values


def get_tfidf_features(load_cached_data=True):
    """
    Generates or loads TF-IDF features for the linear model branch.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val)
    """
    # Define file paths
    train_feat_path = os.path.join(Config.working_dir, "tfidf_train.npz")
    val_feat_path = os.path.join(Config.working_dir, "tfidf_val.npz")
    test_feat_path = os.path.join(Config.working_dir, "tfidf_test.npz")
    train_labels_path = os.path.join(Config.working_dir, "train_labels.npy")
    val_labels_path = os.path.join(Config.working_dir, "val_labels.npy")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data:
        if (
            os.path.exists(train_feat_path)
            and os.path.exists(val_feat_path)
            and os.path.exists(test_feat_path)
            and os.path.exists(train_labels_path)
            and os.path.exists(val_labels_path)
        ):

            print("Loading cached TF-IDF features from disk...")
            X_train = sparse.load_npz(train_feat_path)
            X_val = sparse.load_npz(val_feat_path)
            X_test = sparse.load_npz(test_feat_path)
            y_train = np.load(train_labels_path)
            y_val = np.load(val_labels_path)

            return X_train, X_val, X_test, y_train, y_val
        else:
            print("Cached files not found. Computing features from scratch...")

    # 2. Compute features
    print("Loading raw text data...")
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    if Config.debug:
        print("Debug Mode: Subsampling data for TF-IDF generation.")
        train_df = train_df.sample(
            n=min(1000, len(train_df)), random_state=Config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(500, len(val_df)), random_state=Config.seed
        ).reset_index(drop=True)
        test_df = test_df.iloc[: min(1000, len(test_df))]

    # Handle missing values
    train_text = train_df["comment_text"].fillna("fillna")
    val_text = val_df["comment_text"].fillna("fillna")
    test_text = test_df["comment_text"].fillna("fillna")

    # Concatenate for vocabulary fitting
    all_text = pd.concat([train_text, val_text, test_text])

    print(f"Fitting TF-IDF Vectorizers on {len(all_text)} documents...")

    # Word Vectorizer
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        stop_words="english",
        ngram_range=Config.tfidf_word_ngram_range,
        max_features=Config.tfidf_max_features_word,
    )

    # Char Vectorizer
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="char",
        ngram_range=Config.tfidf_char_ngram_range,
        max_features=Config.tfidf_max_features_char,
    )

    word_vectorizer.fit(all_text)
    char_vectorizer.fit(all_text)

    print("Transforming datasets...")
    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # Stack features
    X_train = sparse.hstack([train_word, train_char])
    X_val = sparse.hstack([val_word, val_char])
    X_test = sparse.hstack([test_word, test_char])

    y_train = train_df[Config.target_cols].values
    y_val = val_df[Config.target_cols].values

    # 3. Save to cache
    print("Saving features to cache...")
    sparse.save_npz(train_feat_path, X_train)
    sparse.save_npz(val_feat_path, X_val)
    sparse.save_npz(test_feat_path, X_test)
    np.save(train_labels_path, y_train)
    np.save(val_labels_path, y_val)

    return X_train, X_val, X_test, y_train, y_val
