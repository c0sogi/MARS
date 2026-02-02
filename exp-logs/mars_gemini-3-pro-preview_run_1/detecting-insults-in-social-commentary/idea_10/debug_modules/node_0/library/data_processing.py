import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


def load_data(debug=Config.DEBUG):
    """
    Loads train, validation, and test datasets from metadata CSVs.
    Handles debug sampling if enabled in Config.
    """
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Handle potential missing values in text columns
    train_df["Comment"] = train_df["Comment"].fillna("")
    val_df["Comment"] = val_df["Comment"].fillna("")
    test_df["Comment"] = test_df["Comment"].fillna("")

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(
            f"[DEBUG] Loaded subset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

    return train_df, val_df, test_df


class SVDFeatureExtractor:
    """
    Extracts structural features using TF-IDF on Word and Char N-grams,
    reduced via TruncatedSVD.
    """

    def __init__(self, n_components=256):
        self.n_components = n_components
        self.pipeline = None

    def fit(self, texts):
        """
        Fits the TF-IDF and SVD pipeline on the provided texts.
        """
        seed_everything(Config.SEED)

        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000,
            sublinear_tf=True,
        )

        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=10000,
            sublinear_tf=True,
        )

        self.pipeline = Pipeline(
            [
                (
                    "union",
                    FeatureUnion(
                        [
                            ("word", word_vectorizer),
                            ("char", char_vectorizer),
                        ]
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=self.n_components, random_state=Config.SEED
                    ),
                ),
            ]
        )

        self.pipeline.fit(texts)
        return self

    def transform(self, texts):
        """
        Transforms texts into SVD feature vectors.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline must be fitted before transform.")
        return self.pipeline.transform(texts).astype(np.float32)


def get_structural_features(
    train_texts, val_texts, test_texts, load_cached_data=True, debug=Config.DEBUG
):
    """
    Generates or loads cached SVD structural features.
    Ensures the pipeline is fitted ONLY on training data to prevent leakage.
    """
    # Define cache filenames
    suffix = "_debug" if debug else ""
    cache_files = {
        "train": os.path.join(Config.CACHE_DIR, f"train_svd{suffix}.npy"),
        "val": os.path.join(Config.CACHE_DIR, f"val_svd{suffix}.npy"),
        "test": os.path.join(Config.CACHE_DIR, f"test_svd{suffix}.npy"),
    }

    # Check cache
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_files.values()):
            print("Loading cached SVD features...")
            return (
                np.load(cache_files["train"]),
                np.load(cache_files["val"]),
                np.load(cache_files["test"]),
            )
        else:
            print("Cache missing or incomplete. Recomputing features...")

    # Compute features
    print("Fitting SVD Feature Extractor on Training Data...")
    extractor = SVDFeatureExtractor(n_components=Config.SVD_COMPONENTS)
    extractor.fit(train_texts)

    print("Transforming datasets...")
    train_svd = extractor.transform(train_texts)
    val_svd = extractor.transform(val_texts)
    test_svd = extractor.transform(test_texts)

    # Save to cache
    print("Saving SVD features to cache...")
    np.save(cache_files["train"], train_svd)
    np.save(cache_files["val"], val_svd)
    np.save(cache_files["test"], test_svd)

    return train_svd, val_svd, test_svd


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles Tokenization and SVD Feature retrieval.
    """

    def __init__(
        self, texts, svd_features, labels=None, tokenizer=None, max_len=Config.MAX_LEN
    ):
        self.texts = texts
        self.svd_features = svd_features
        self.labels = labels
        self.max_len = max_len

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        svd_vec = self.svd_features[idx]

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
            "svd_features": torch.tensor(svd_vec, dtype=torch.float32),
        }

        if self.labels is not None:
            # Labels can be int (0/1) or float (soft labels)
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def prepare_student_data(train_df, test_df, test_soft_labels, svd_train, svd_test):
    """
    Combines labeled training data and soft-labeled test data for the Student model.

    Args:
        train_df: DataFrame containing training data with 'Comment' and 'Insult'.
        test_df: DataFrame containing test data with 'Comment'.
        test_soft_labels: Numpy array of soft probabilities for test data.
        svd_train: SVD features for training data.
        svd_test: SVD features for test data.

    Returns:
        combined_texts: List of all comments.
        combined_svd: Numpy array of all SVD features.
        combined_labels: Numpy array of labels (hard 0/1 for train, soft probs for test).
    """
    # Extract texts
    train_texts = train_df["Comment"].tolist()
    test_texts = test_df["Comment"].tolist()
    combined_texts = train_texts + test_texts

    # Combine SVD features
    combined_svd = np.vstack([svd_train, svd_test])

    # Combine labels
    # Ensure train labels are float for consistency with soft labels
    train_labels = train_df["Insult"].values.astype(np.float32)
    combined_labels = np.concatenate([train_labels, test_soft_labels])

    return combined_texts, combined_svd, combined_labels
