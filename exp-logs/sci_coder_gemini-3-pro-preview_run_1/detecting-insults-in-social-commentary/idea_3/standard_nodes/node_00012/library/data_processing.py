import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
from transformers import AutoTokenizer, logging
from library.config import Config
from library.utils import set_seed

# Suppress verbose warnings from transformers
logging.set_verbosity_error()


class StructuralFeatureExtractor:
    """
    Extracts structural features using Character N-grams and SVD.
    This captures morphological patterns and obfuscated text often found in insults.
    """

    def __init__(
        self,
        ngram_range=None,
        word_ngram_range=None,
        max_features=None,
        n_components=None,
        seed=None,
    ):
        char_ngram_range = (
            ngram_range if ngram_range is not None else Config.NGRAM_RANGE
        )
        word_ngram_range = (
            word_ngram_range
            if word_ngram_range is not None
            else Config.WORD_NGRAM_RANGE
        )
        max_features = (
            max_features if max_features is not None else Config.TFIDF_MAX_FEATURES
        )
        n_components = (
            n_components if n_components is not None else Config.SVD_COMPONENTS
        )
        seed = seed if seed is not None else Config.SEED

        # Cite solution_lesson_node_00002: Dual TF-IDF (Word + Char)
        # Cite solution_lesson_node_00004: Semantic sweet spot (Word 1-2, Char 3-5)
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            max_features=max_features,
            min_df=2,
            use_idf=True,
            sublinear_tf=True,
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            max_features=max_features,
            min_df=2,
            use_idf=True,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(
            n_components=n_components, random_state=seed, algorithm="randomized"
        )

    def fit(self, texts):
        """
        Fits the vectorizers and SVD on the provided texts.
        """
        # Fit TF-IDF
        char_tfidf = self.char_vectorizer.fit_transform(texts)
        word_tfidf = self.word_vectorizer.fit_transform(texts)

        # Combine features
        combined_tfidf = hstack([char_tfidf, word_tfidf])

        # Fit SVD on the combined matrix
        # Cite solution_lesson_node_00011: Project high-dimensional sparse features into dense space
        self.svd.fit(combined_tfidf)
        return self

    def transform(self, texts):
        """
        Transforms texts into dense SVD vectors.
        """
        char_tfidf = self.char_vectorizer.transform(texts)
        word_tfidf = self.word_vectorizer.transform(texts)
        combined_tfidf = hstack([char_tfidf, word_tfidf])

        dense_features = self.svd.transform(combined_tfidf)
        return dense_features


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Combines tokenized text with structural features.
    """

    def __init__(
        self,
        df,
        structural_features,
        tokenizer,
        max_length=Config.MAX_LENGTH,
        is_test=False,
    ):
        # Ensure text is string and handle NaNs
        self.texts = df["Comment"].fillna("").astype(str).tolist()
        self.structural_features = structural_features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        if not is_test:
            self.labels = df["Insult"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Get structural features as float tensor
        struct_feat = torch.tensor(self.structural_features[idx], dtype=torch.float32)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "structural_features": struct_feat,
        }

        if not self.is_test:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            item["label"] = label

        return item


def get_structural_features(train_texts, val_texts, test_texts, load_cached_data=True):
    """
    Computes or loads cached structural features.
    Uses .npy files for caching to avoid pickle.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Namespace cache artifacts by configuration (Cite debug_lesson_2)
    suffix = f"_{Config.SVD_COMPONENTS}"
    train_feat_path = os.path.join(cache_dir, f"train_struct{suffix}.npy")
    val_feat_path = os.path.join(cache_dir, f"val_struct{suffix}.npy")
    test_feat_path = os.path.join(cache_dir, f"test_struct{suffix}.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_feat_path)
        and os.path.exists(val_feat_path)
        and os.path.exists(test_feat_path)
    )

    # Flag to track if valid cache was loaded
    loaded_valid_cache = False

    if load_cached_data and cache_exists:
        print(f"Loading cached structural features from {cache_dir}...")
        try:
            train_features = np.load(train_feat_path)
            val_features = np.load(val_feat_path)
            test_features = np.load(test_feat_path)

            # Validate Cache Integrity (Cite debug_lesson_1)
            if train_features.shape[1] == Config.SVD_COMPONENTS:
                loaded_valid_cache = True
            else:
                print(
                    f"Dimension mismatch: Cached {train_features.shape[1]}, "
                    f"Expected {Config.SVD_COMPONENTS}. Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    if not loaded_valid_cache:
        # Recompute features
        extractor = StructuralFeatureExtractor()

        # Fit only on training data to prevent leakage
        extractor.fit(train_texts)

        # Transform all splits
        train_features = extractor.transform(train_texts)
        val_features = extractor.transform(val_texts)
        test_features = extractor.transform(test_texts)

        # Save to cache
        np.save(train_feat_path, train_features)
        np.save(val_feat_path, val_features)
        np.save(test_feat_path, test_features)

    return train_features, val_features, test_features


def load_data(load_cached_data=True):
    """
    Main function to load data, process features, and create Datasets.

    Args:
        load_cached_data (bool): Whether to try loading cached structural features.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    set_seed()

    # Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Extract raw texts for feature extraction
    train_texts = train_df["Comment"].fillna("").astype(str).tolist()
    val_texts = val_df["Comment"].fillna("").astype(str).tolist()
    test_texts = test_df["Comment"].fillna("").astype(str).tolist()

    # Get Structural Features (Cached or Computed)
    train_struct, val_struct, test_struct = get_structural_features(
        train_texts, val_texts, test_texts, load_cached_data=load_cached_data
    )

    # Initialize Tokenizer
    # use_fast=True is generally recommended, but we can use False if dependencies are tricky.
    # Given the environment, we'll stick to default (Fast if available).
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = InsultDataset(train_df, train_struct, tokenizer, is_test=False)
    val_dataset = InsultDataset(val_df, val_struct, tokenizer, is_test=False)
    test_dataset = InsultDataset(test_df, test_struct, tokenizer, is_test=True)

    return train_dataset, val_dataset, test_dataset
