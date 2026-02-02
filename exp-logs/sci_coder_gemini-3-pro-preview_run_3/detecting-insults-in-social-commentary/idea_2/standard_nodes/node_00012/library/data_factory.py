import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from library.utils import set_seed


class InsultDataset(Dataset):
    """
    PyTorch Dataset for the Neural Stream (Transformer-based).
    Handles tokenization and tensor creation.
    """

    def __init__(self, df, tokenizer, max_len=128):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'Comment' and optionally 'Insult'.
            tokenizer: Transformer tokenizer instance (e.g., RoBERTa tokenizer).
            max_len (int): Maximum sequence length for truncation/padding.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.df.loc[index, "Comment"])

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # Add target if available (Training/Validation mode)
        if "Insult" in self.df.columns:
            target = self.df.loc[index, "Insult"]
            item["target"] = torch.tensor(target, dtype=torch.float)

        return item


def get_tfidf_features(
    train_df, val_df, test_df, load_cached_data=True, cache_dir="./working/idea_2"
):
    """
    Generates and caches TF-IDF features for the Statistical Stream (Linear Model).

    Args:
        train_df, val_df, test_df: DataFrames for the respective splits.
        load_cached_data (bool): If True, attempts to load from disk.
        cache_dir (str): Directory to store/load .npy files.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test) as numpy arrays.
    """
    set_seed(42)
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    files = {
        "X_train": "tfidf_X_train.npy",
        "y_train": "tfidf_y_train.npy",
        "X_val": "tfidf_X_val.npy",
        "y_val": "tfidf_y_val.npy",
        "X_test": "tfidf_X_test.npy",
    }

    # Check if all cache files exist
    all_exist = all(os.path.exists(os.path.join(cache_dir, f)) for f in files.values())

    if load_cached_data and all_exist:
        print("Loading cached TF-IDF features from disk...")
        data = {}
        for k, v in files.items():
            data[k] = np.load(os.path.join(cache_dir, v))
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
        )

    print("Computing TF-IDF features from scratch...")

    # Prepare text data (ensure strings)
    train_text = train_df["Comment"].fillna("").astype(str)
    val_text = val_df["Comment"].fillna("").astype(str)
    test_text = test_df["Comment"].fillna("").astype(str)

    # Prepare targets
    y_train = (
        train_df["Insult"].values.astype(np.float32)
        if "Insult" in train_df.columns
        else np.zeros(len(train_df), dtype=np.float32)
    )
    y_val = (
        val_df["Insult"].values.astype(np.float32)
        if "Insult" in val_df.columns
        else np.zeros(len(val_df), dtype=np.float32)
    )

    # 1. Word N-grams (1-3)
    print("Fitting Word Vectorizer (1-3 grams)...")
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer="word",
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    word_vectorizer.fit(train_text)

    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    # 2. Character N-grams (3-5) - skipping noisy 2-grams
    print("Fitting Character Vectorizer (3-5 grams)...")
    char_vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char",
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vectorizer.fit(train_text)

    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # 3. Stack Features
    print("Stacking features...")
    X_train_sparse = hstack([train_word, train_char])
    X_val_sparse = hstack([val_word, val_char])
    X_test_sparse = hstack([test_word, test_char])

    # 4. Convert to Dense (required for .npy saving)
    # Note: With ~3k samples and ~50k features, dense size is manageable (~600MB)
    print("Converting to dense arrays...")
    X_train = X_train_sparse.toarray().astype(np.float32)
    X_val = X_val_sparse.toarray().astype(np.float32)
    X_test = X_test_sparse.toarray().astype(np.float32)

    # 5. Save to Cache
    print(f"Saving features to {cache_dir}...")
    np.save(os.path.join(cache_dir, files["X_train"]), X_train)
    np.save(os.path.join(cache_dir, files["y_train"]), y_train)
    np.save(os.path.join(cache_dir, files["X_val"]), X_val)
    np.save(os.path.join(cache_dir, files["y_val"]), y_val)
    np.save(os.path.join(cache_dir, files["X_test"]), X_test)

    return X_train, y_train, X_val, y_val, X_test
