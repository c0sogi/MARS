import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from transformers import AutoTokenizer
from library.config import Config
from library.utils import save_parquet, load_parquet


class TextDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Transformer models.
    Handles tokenization and label encoding.
    """

    def __init__(
        self,
        texts,
        labels=None,
        tokenizer_name=Config.MODEL_DEBERTA,
        max_len=Config.MAX_LEN,
    ):
        """
        Args:
            texts (list or pd.Series): Input text sequences.
            labels (list or pd.Series, optional): Target labels (EAP, HPL, MWS).
            tokenizer_name (str): HuggingFace model name for tokenizer.
            max_len (int): Maximum sequence length.
        """
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.label_map = {"EAP": 0, "HPL": 1, "MWS": 2}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Extract tensors (remove batch dimension added by tokenizer)
        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)

        item = {"input_ids": ids, "attention_mask": mask}

        # Add label if available
        if self.labels is not None:
            label_str = self.labels[idx]
            label_id = self.label_map[label_str]
            item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item


def load_data(
    load_cached_data=True, debug=Config.DEBUG, sample_size=Config.DEBUG_SAMPLE_SIZE
):
    """
    Loads data from metadata, merges train/val for CV, creates folds, and caches result.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): Whether to use a small subset for debugging.
        sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (df_train, df_test)
            df_train: DataFrame with 'text', 'author', 'fold' columns.
            df_test: DataFrame with 'text', 'id' columns.
    """
    # Define cache filenames
    train_cache_name = "train_folds_debug.parquet" if debug else "train_folds.parquet"
    test_cache_name = "test_data_debug.parquet" if debug else "test_data.parquet"

    # 1. Try to load from cache
    if load_cached_data:
        df_train = load_parquet(train_cache_name)
        df_test = load_parquet(test_cache_name)

        if df_train is not None and df_test is not None:
            return df_train, df_test

    # 2. Load from Metadata
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_META_PATH}")

    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # 3. Merge Train and Val for full Cross-Validation
    df_train = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(drop=True)

    # 4. Handle Debugging (Downsampling)
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # 5. Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df_train["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train["author"])):
        df_train.loc[val_idx, "fold"] = fold

    # 6. Cache the processed data
    save_parquet(df_train, train_cache_name)
    save_parquet(df_test, test_cache_name)

    return df_train, df_test


def get_classical_features(train_texts, val_texts, test_texts=None):
    """
    Generates TF-IDF and SVD features for classical models.
    Fits on train_texts, transforms val_texts and test_texts.

    Args:
        train_texts (iterable): Training text samples.
        val_texts (iterable): Validation text samples.
        test_texts (iterable, optional): Test text samples.

    Returns:
        tuple: (train_tfidf, val_tfidf, [test_tfidf], train_svd, val_svd, [test_svd])
               TF-IDF matrices are sparse, SVD matrices are dense.
    """
    # 1. Define Vectorizers
    # Word N-grams (1-3)
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=Config.TFIDF_MIN_DF,
        analyzer="word",
        token_pattern=r"\w{1,}",
        sublinear_tf=True,
    )

    # Character N-grams (2-5)
    char_vectorizer = TfidfVectorizer(
        ngram_range=(2, 5),
        min_df=Config.TFIDF_MIN_DF,
        analyzer="char",
        sublinear_tf=True,
    )

    # Combine Word and Char features
    vectorizer = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    # 2. Fit and Transform (Sparse Features)
    # Fit only on training data to prevent leakage
    train_tfidf = vectorizer.fit_transform(train_texts)
    val_tfidf = vectorizer.transform(val_texts)

    # 3. SVD Projection (Dense Features for XGBoost)
    svd = TruncatedSVD(n_components=Config.SVD_COMPONENTS, random_state=Config.SEED)
    train_svd = svd.fit_transform(train_tfidf)
    val_svd = svd.transform(val_tfidf)

    # 4. Handle Test Data if provided
    if test_texts is not None:
        test_tfidf = vectorizer.transform(test_texts)
        test_svd = svd.transform(test_tfidf)
        return train_tfidf, val_tfidf, test_tfidf, train_svd, val_svd, test_svd

    return train_tfidf, val_tfidf, train_svd, val_svd
