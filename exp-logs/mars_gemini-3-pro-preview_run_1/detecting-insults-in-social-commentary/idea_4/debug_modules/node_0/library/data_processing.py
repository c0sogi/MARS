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


class FeatureEngineer:
    """
    Handles the generation of structural features using TF-IDF and TruncatedSVD.
    """

    def __init__(self, svd_components=256, seed=42):
        self.svd_components = svd_components
        self.seed = seed

        # Define the TF-IDF vectorizers for Word and Char n-grams
        self.vectorizer = FeatureUnion(
            [
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        analyzer="word",
                        min_df=3,
                        token_pattern=r"\w{1,}",
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=3),
                ),
            ]
        )

        # Define TruncatedSVD
        self.svd = TruncatedSVD(
            n_components=self.svd_components, random_state=self.seed
        )

        # Combine into a pipeline
        self.pipeline = Pipeline([("vectorizer", self.vectorizer), ("svd", self.svd)])

        self.is_fitted = False

    def fit(self, texts):
        """Fits the pipeline on the provided texts."""
        print("Fitting TF-IDF and SVD pipeline...")
        self.pipeline.fit(texts)
        self.is_fitted = True

    def transform(self, texts):
        """Transforms texts into dense SVD features."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform.")
        print("Transforming texts to SVD features...")
        return self.pipeline.transform(texts).astype(np.float32)


class InsultDataset(Dataset):
    """
    PyTorch Dataset for the Insult Detection task.
    Returns tokenized inputs for DeBERTa and structural SVD features.
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
        svd_feat = self.svd_features[idx]

        # Tokenize raw text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by tokenizer
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "svd_feat": torch.tensor(svd_feat, dtype=torch.float32),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def get_datasets(load_cached_data=True, debug=False):
    """
    Main data processing function.
    Loads data, generates/loads features, and returns PyTorch Datasets.

    Args:
        load_cached_data (bool): If True, attempts to load features from disk.
        debug (bool): If True, uses a small subset of data.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    seed_everything(Config.seed)

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "train": os.path.join(Config.working_dir, "train_svd.npy"),
        "val": os.path.join(Config.working_dir, "val_svd.npy"),
        "test": os.path.join(Config.working_dir, "test_svd.npy"),
    }

    # Load raw metadata
    print(f"Loading metadata from {Config.metadata_dir}...")
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # Handle missing text
    df_train["Comment"] = df_train["Comment"].fillna("")
    df_val["Comment"] = df_val["Comment"].fillna("")
    df_test["Comment"] = df_test["Comment"].fillna("")

    # Debug mode: subsample
    if debug:
        print("DEBUG MODE: Subsampling data...")
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    # Check cache availability
    cache_exists = all(os.path.exists(p) for p in cache_paths.values())

    features = {}

    if load_cached_data and cache_exists and not debug:
        print("Loading cached SVD features...")
        features["train"] = np.load(cache_paths["train"])
        features["val"] = np.load(cache_paths["val"])
        features["test"] = np.load(cache_paths["test"])
    else:
        print("Generating SVD features from scratch...")
        fe = FeatureEngineer(svd_components=Config.svd_components, seed=Config.seed)

        # Fit on training data
        fe.fit(df_train["Comment"].tolist())

        # Transform all sets
        features["train"] = fe.transform(df_train["Comment"].tolist())
        features["val"] = fe.transform(df_val["Comment"].tolist())
        features["test"] = fe.transform(df_test["Comment"].tolist())

        # Cache features (only if not in debug mode to avoid overwriting full cache with debug data)
        if not debug:
            print(f"Saving SVD features to {Config.working_dir}...")
            np.save(cache_paths["train"], features["train"])
            np.save(cache_paths["val"], features["val"])
            np.save(cache_paths["test"], features["test"])

    # Initialize Tokenizer
    print(f"Initializing tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    print("Creating PyTorch Datasets...")
    train_dataset = InsultDataset(
        texts=df_train["Comment"].values,
        svd_features=features["train"],
        labels=df_train["Insult"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
    )

    val_dataset = InsultDataset(
        texts=df_val["Comment"].values,
        svd_features=features["val"],
        labels=df_val["Insult"].values,
        tokenizer=tokenizer,
        max_len=Config.max_len,
    )

    test_dataset = InsultDataset(
        texts=df_test["Comment"].values,
        svd_features=features["test"],
        labels=None,  # Test set has no labels
        tokenizer=tokenizer,
        max_len=Config.max_len,
    )

    print(f"Data processing complete.")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, tokenizer
