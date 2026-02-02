import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Tuple, List, Optional, Dict, Union

from library.config import Config
from library.utils import save_npy, load_npy, seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Returns tokenized text features and structural SVD features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        svd_features: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = Config.MAX_LEN,
        is_test: bool = False,
    ):
        self.texts = df["Comment"].fillna("").astype(str).values
        self.svd_features = svd_features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            self.labels = df["Insult"].values
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        # Structural Features
        svd_feat = torch.tensor(self.svd_features[idx], dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "svd_feat": svd_feat,
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def get_tokenizer() -> PreTrainedTokenizerBase:
    """
    Loads the tokenizer for the configured model.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_NAME)


def apply_layer_norm(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Applies Layer Normalization (row-wise) to numpy array.
    Normalizes each sample to have mean 0 and variance 1.
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / (std + epsilon)


def compute_structural_features(
    train_text: List[str], val_text: List[str], test_text: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes TF-IDF (Word 1-2gram, Char 3-5gram) -> SVD -> LayerNorm.
    Fits ONLY on train_text. Transforms val_text and test_text.
    """
    print("Extracting TF-IDF features (Word 1-2 grams)...")
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=None,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        stop_words="english",
    )

    print("Extracting TF-IDF features (Char 3-5 grams)...")
    char_vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        min_df=3,
        max_features=None,
        strip_accents="unicode",
        analyzer="char",
        sublinear_tf=True,
    )

    # Fit on Train only
    print("Fitting vectorizers on training data...")
    train_word = word_vectorizer.fit_transform(train_text)
    train_char = char_vectorizer.fit_transform(train_text)

    # Transform others
    print("Transforming validation and test data...")
    val_word = word_vectorizer.transform(val_text)
    val_char = char_vectorizer.transform(val_text)

    test_word = word_vectorizer.transform(test_text)
    test_char = char_vectorizer.transform(test_text)

    # Stack features
    train_sparse = hstack([train_word, train_char])
    val_sparse = hstack([val_word, val_char])
    test_sparse = hstack([test_word, test_char])

    # SVD
    print(f"Fitting TruncatedSVD (dim={Config.SVD_DIM}) on training data...")
    svd = TruncatedSVD(n_components=Config.SVD_DIM, random_state=Config.SEED)
    train_svd = svd.fit_transform(train_sparse)
    val_svd = svd.transform(val_sparse)
    test_svd = svd.transform(test_sparse)

    # Layer Normalization
    print("Applying Layer Normalization to SVD features...")
    train_svd = apply_layer_norm(train_svd)
    val_svd = apply_layer_norm(val_svd)
    test_svd = apply_layer_norm(test_svd)

    return train_svd, val_svd, test_svd


def load_data(
    load_cached_data: bool = True, debug: bool = False
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Main function to load data and features.
    Handles caching of SVD features to avoid re-computation.

    Args:
        load_cached_data: If True, attempts to load .npy files from working dir.
        debug: If True, truncates datasets for quick testing.

    Returns:
        (train_df, val_df, test_df, train_svd, val_svd, test_svd)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_svd_path = os.path.join(cache_dir, "train_svd.npy")
    val_svd_path = os.path.join(cache_dir, "val_svd.npy")
    test_svd_path = os.path.join(cache_dir, "test_svd.npy")

    # Load Raw Data
    print("Loading raw CSV data...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if debug:
        print("DEBUG MODE: Truncating datasets...")
        train_df = train_df.iloc[:100].reset_index(drop=True)
        val_df = val_df.iloc[:50].reset_index(drop=True)
        test_df = test_df.iloc[:50].reset_index(drop=True)

    # Check Cache
    cache_exists = (
        os.path.exists(train_svd_path)
        and os.path.exists(val_svd_path)
        and os.path.exists(test_svd_path)
    )

    if load_cached_data and cache_exists:
        print("Loading cached SVD features...")
        train_svd = load_npy(train_svd_path)
        val_svd = load_npy(val_svd_path)
        test_svd = load_npy(test_svd_path)

        # Verify shapes match current data (important for debug mode toggling)
        if len(train_svd) != len(train_df):
            print(
                "Cache shape mismatch (likely due to debug mode change). Recomputing..."
            )
            cache_exists = False

    if not (load_cached_data and cache_exists):
        print("Computing structural features from scratch...")

        # Fill NaNs
        train_text = train_df["Comment"].fillna("").astype(str).tolist()
        val_text = val_df["Comment"].fillna("").astype(str).tolist()
        test_text = test_df["Comment"].fillna("").astype(str).tolist()

        train_svd, val_svd, test_svd = compute_structural_features(
            train_text, val_text, test_text
        )

        # Save to cache
        print("Saving SVD features to cache...")
        save_npy(train_svd, train_svd_path)
        save_npy(val_svd, val_svd_path)
        save_npy(test_svd, test_svd_path)

    return train_df, val_df, test_df, train_svd, val_svd, test_svd


def create_dataloader(
    df: pd.DataFrame,
    svd_features: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = Config.BATCH_SIZE,
    is_train: bool = True,
    is_test: bool = False,
    shuffle: bool = True,
    num_workers: int = Config.NUM_WORKERS,
) -> DataLoader:
    """
    Factory function to create a DataLoader.
    """
    dataset = InsultDataset(
        df=df,
        svd_features=svd_features,
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN,
        is_test=is_test,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(is_train and len(df) > batch_size),
    )
