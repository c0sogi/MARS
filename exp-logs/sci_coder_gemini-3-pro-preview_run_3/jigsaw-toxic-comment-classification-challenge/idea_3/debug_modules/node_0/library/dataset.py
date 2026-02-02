import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from library.config import Config
from library.utils import seed_everything


def load_data_splits():
    """
    Loads train, val, and test dataframes using metadata to ensure correct splits.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_CSV)
    val_meta = pd.read_csv(Config.VAL_META_CSV)
    test_meta = pd.read_csv(Config.TEST_META_CSV)

    # Load Raw Data
    # We read the full files once.
    # Note: train.csv contains labels, but metadata is the source of truth for the split.
    df_train_raw = pd.read_csv(Config.TRAIN_CSV)
    df_test_raw = pd.read_csv(Config.TEST_CSV)

    # --- Prepare Train Split ---
    # Map metadata indices to raw dataframe
    train_indices = train_meta["source_row_index"].values
    df_train = df_train_raw.iloc[train_indices].copy().reset_index(drop=True)
    # Ensure labels match metadata (drop raw labels and merge from metadata)
    df_train = df_train.drop(columns=Config.LABEL_COLS, errors="ignore")
    # We use the index or ID to align, but since we sliced by source_row_index,
    # the order should align with metadata if we didn't shuffle metadata.
    # To be safe, we reset index on both and concat, assuming metadata is sorted/aligned by design.
    # A safer approach is to merge on ID.
    df_train = df_train.drop(
        columns=["id"], errors="ignore"
    )  # Drop ID to avoid collision
    df_train = pd.concat([train_meta, df_train[["comment_text"]]], axis=1)

    # --- Prepare Val Split ---
    val_indices = val_meta["source_row_index"].values
    df_val = df_train_raw.iloc[val_indices].copy().reset_index(drop=True)
    df_val = df_val.drop(columns=Config.LABEL_COLS, errors="ignore")
    df_val = df_val.drop(columns=["id"], errors="ignore")
    df_val = pd.concat([val_meta, df_val[["comment_text"]]], axis=1)

    # --- Prepare Test Split ---
    test_indices = test_meta["source_row_index"].values
    df_test = df_test_raw.iloc[test_indices].copy().reset_index(drop=True)
    # Test metadata usually just maps IDs. We merge text.
    df_test = df_test.drop(columns=["id"], errors="ignore")
    df_test = pd.concat([test_meta, df_test[["comment_text"]]], axis=1)

    return df_train, df_val, df_test


class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=Config.MAX_LEN, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.text = df["comment_text"].fillna("").astype(str).values

        if not self.is_test:
            self.labels = df[Config.LABEL_COLS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]

        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)
        token_type_ids = inputs["token_type_ids"].squeeze(0)

        data = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": token_type_ids,
        }

        if not self.is_test:
            labels = torch.tensor(self.labels[index], dtype=torch.float)
            data["labels"] = labels

        return data


def get_nbsvm_features(df_train, df_val, df_test, load_cached_data=True):
    """
    Generates or loads TF-IDF features for NBSVM.

    Args:
        df_train, df_val, df_test: DataFrames containing 'comment_text'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, X_val, X_test) as sparse matrices.
    """
    # Define cache paths (Validation paths are derived relative to working dir)
    path_word_train = Config.CACHE_NBSVM_WORD_TRAIN
    path_word_val = os.path.join(Config.WORKING_DIR, "nbsvm_word_val.npz")
    path_word_test = Config.CACHE_NBSVM_WORD_TEST

    path_char_train = Config.CACHE_NBSVM_CHAR_TRAIN
    path_char_val = os.path.join(Config.WORKING_DIR, "nbsvm_char_val.npz")
    path_char_test = Config.CACHE_NBSVM_CHAR_TEST

    cache_files = [
        path_word_train,
        path_word_val,
        path_word_test,
        path_char_train,
        path_char_val,
        path_char_test,
    ]

    # 1. Attempt to load from cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files):
        print("Loading cached NBSVM features...")
        try:
            train_word = sparse.load_npz(path_word_train)
            val_word = sparse.load_npz(path_word_val)
            test_word = sparse.load_npz(path_word_test)

            train_char = sparse.load_npz(path_char_train)
            val_char = sparse.load_npz(path_char_val)
            test_char = sparse.load_npz(path_char_test)

            X_train = sparse.hstack([train_word, train_char])
            X_val = sparse.hstack([val_word, val_char])
            X_test = sparse.hstack([test_word, test_char])
            return X_train, X_val, X_test
        except Exception as e:
            print(f"Error loading cache ({e}). Recomputing features...")

    # 2. Compute features
    print("Computing NBSVM features...")

    train_text = df_train["comment_text"].fillna("")
    val_text = df_val["comment_text"].fillna("")
    test_text = df_test["comment_text"].fillna("")

    # Fit on Train + Val to ensure vocabulary coverage while avoiding test leakage
    fit_text = pd.concat([train_text, val_text])

    # Word Vectorizer
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.NBSVM_WORD_NGRAM_RANGE,
        min_df=Config.NBSVM_MIN_DF,
        strip_accents="unicode",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )

    # Char Vectorizer
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.NBSVM_CHAR_NGRAM_RANGE,
        min_df=Config.NBSVM_MIN_DF,
        strip_accents="unicode",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        analyzer="char",
    )

    print("Fitting Word Vectorizer...")
    word_vectorizer.fit(fit_text)
    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    print("Fitting Char Vectorizer...")
    char_vectorizer.fit(fit_text)
    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    sparse.save_npz(path_word_train, train_word)
    sparse.save_npz(path_word_val, val_word)
    sparse.save_npz(path_word_test, test_word)

    sparse.save_npz(path_char_train, train_char)
    sparse.save_npz(path_char_val, val_char)
    sparse.save_npz(path_char_test, test_char)

    # Stack features
    X_train = sparse.hstack([train_word, train_char])
    X_val = sparse.hstack([val_word, val_char])
    X_test = sparse.hstack([test_word, test_char])

    return X_train, X_val, X_test
