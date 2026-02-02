import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from library.config import Config
from library.utils import seed_everything


def load_data():
    """
    Loads the train, validation, and test metadata CSVs.
    """
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Fill NaNs in Comment column to avoid issues during vectorization
    train_df["Comment"] = train_df["Comment"].fillna("")
    val_df["Comment"] = val_df["Comment"].fillna("")
    test_df["Comment"] = test_df["Comment"].fillna("")

    return train_df, val_df, test_df


def get_structural_features(train_texts, val_texts, test_texts, load_cached_data=True):
    """
    Generates or loads structural features using TF-IDF and TruncatedSVD.
    Fits only on train_texts. Transforms val_texts and test_texts.

    Args:
        train_texts (list): List of training text strings.
        val_texts (list): List of validation text strings.
        test_texts (list): List of test text strings.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (train_svd, val_svd, test_svd) as numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_svd.npy")
    val_cache_path = os.path.join(cache_dir, "val_svd.npy")
    test_cache_path = os.path.join(cache_dir, "test_svd.npy")

    # Check if cache exists
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading structural features from cache...")
            train_svd = np.load(train_cache_path)
            val_svd = np.load(val_cache_path)
            test_svd = np.load(test_cache_path)
            return train_svd, val_svd, test_svd
        else:
            print("Cache not found. Computing structural features...")
    else:
        print("Forcing re-computation of structural features...")

    # Define Feature Extraction Pipeline
    # Word N-grams
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=Config.NGRAM_RANGE_WORD,
        min_df=2,
        sublinear_tf=True,
    )

    # Char N-grams
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=Config.NGRAM_RANGE_CHAR,
        min_df=2,
        sublinear_tf=True,
    )

    # Combine features
    vectorizer = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])

    # Dimensionality Reduction
    # We fit the vectorizer first to get sparse matrices, then SVD
    # Or use a Pipeline. Using Pipeline for cleanliness.
    pipeline = Pipeline(
        [
            ("vect", vectorizer),
            (
                "svd",
                TruncatedSVD(
                    n_components=Config.SVD_COMPONENTS, random_state=Config.SEED
                ),
            ),
        ]
    )

    print("Fitting vectorizers and SVD on training data...")
    pipeline.fit(train_texts)

    print("Transforming datasets...")
    train_svd = pipeline.transform(train_texts)
    val_svd = pipeline.transform(val_texts)
    test_svd = pipeline.transform(test_texts)

    # Cast to float32 to save space and match torch default
    train_svd = train_svd.astype(np.float32)
    val_svd = val_svd.astype(np.float32)
    test_svd = test_svd.astype(np.float32)

    # Save to cache
    print(f"Saving structural features to {cache_dir}...")
    np.save(train_cache_path, train_svd)
    np.save(val_cache_path, val_svd)
    np.save(test_cache_path, test_svd)

    return train_svd, val_svd, test_svd


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (MLM).
    Dynamically masks tokens during training.
    """

    def __init__(
        self, texts, tokenizer, max_len=Config.MAX_LEN, mask_prob=Config.MLM_MASK_PROB
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        # Create labels for MLM (copy of input_ids)
        labels = input_ids.clone()

        # Create mask array
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        # Do not mask special tokens
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Determine which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Replace masked input tokens with [MASK] token id
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        # We only compute loss on masked tokens. Set others to -100.
        labels[~masked_indices] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InsultDataset(Dataset):
    """
    Dataset for Insult Classification.
    Returns tokenized text, structural features (SVD), and labels.
    """

    def __init__(
        self, texts, svd_features, tokenizer, max_len=Config.MAX_LEN, labels=None
    ):
        self.texts = texts
        self.svd_features = svd_features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        svd_vec = torch.tensor(self.svd_features[idx], dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "svd_features": svd_vec,
        }

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            item["label"] = label

        return item
