import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import AutoTokenizer, logging as hf_logging
from library.utils import generate_config_hash, set_seed

# Suppress transformer warnings
hf_logging.set_verbosity_error()

CACHE_DIR = "./working/idea_3"


class ClassicalFeaturePipeline:
    """
    Manages the creation of classical NLP features:
    1. Sparse TF-IDF (Word n-grams + Char n-grams)
    2. Dense SVD projections
    """

    def __init__(self, config):
        self.config = config
        self.seed = config.get("seed", 42)
        self.svd_n_components = config.get("svd_n_components", 50)

    def execute(self, train_text, test_text, load_cached_data=True):
        """
        Generates or loads classical features for both train and test sets.

        Args:
            train_text (pd.Series or list): Training text data.
            test_text (pd.Series or list): Test text data.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (train_sparse, train_dense, test_sparse, test_dense)
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Generate hash for caching based on config
        config_hash = generate_config_hash(self.config)

        # Define paths
        train_sparse_path = os.path.join(CACHE_DIR, f"train_sparse_{config_hash}.npz")
        test_sparse_path = os.path.join(CACHE_DIR, f"test_sparse_{config_hash}.npz")
        train_dense_path = os.path.join(CACHE_DIR, f"train_dense_{config_hash}.npy")
        test_dense_path = os.path.join(CACHE_DIR, f"test_dense_{config_hash}.npy")

        # Check if all files exist
        if load_cached_data and all(
            os.path.exists(p)
            for p in [
                train_sparse_path,
                test_sparse_path,
                train_dense_path,
                test_dense_path,
            ]
        ):
            print(f"Loading classical features from cache: {CACHE_DIR}")
            train_sparse = scipy.sparse.load_npz(train_sparse_path)
            test_sparse = scipy.sparse.load_npz(test_sparse_path)
            train_dense = np.load(train_dense_path)
            test_dense = np.load(test_dense_path)
            return train_sparse, train_dense, test_sparse, test_dense

        print("Computing classical features from scratch...")
        set_seed(self.seed)

        # 1. TF-IDF Vectorization
        # We keep stopwords for authorship attribution as they contain stylistic signals
        print("  - Fitting Word TF-IDF (1-3 grams)...")
        word_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            token_pattern=r"\w+",
            sublinear_tf=True,
            use_idf=True,
        )
        train_word = word_vec.fit_transform(train_text)
        test_word = word_vec.transform(test_text)

        print("  - Fitting Char TF-IDF (2-5 grams)...")
        char_vec = TfidfVectorizer(
            ngram_range=(2, 5),
            min_df=2,
            analyzer="char",
            sublinear_tf=True,
            use_idf=True,
        )
        train_char = char_vec.fit_transform(train_text)
        test_char = char_vec.transform(test_text)

        # Stack features
        print("  - Stacking sparse features...")
        train_sparse = scipy.sparse.hstack([train_word, train_char], format="csr")
        test_sparse = scipy.sparse.hstack([test_word, test_char], format="csr")

        # 2. Dimensionality Reduction (SVD)
        print(f"  - Fitting TruncatedSVD (n={self.svd_n_components})...")
        svd = TruncatedSVD(n_components=self.svd_n_components, random_state=self.seed)
        train_dense = svd.fit_transform(train_sparse)
        test_dense = svd.transform(test_sparse)

        # Save to cache
        print(f"Saving classical features to {CACHE_DIR}...")
        scipy.sparse.save_npz(train_sparse_path, train_sparse)
        scipy.sparse.save_npz(test_sparse_path, test_sparse)
        np.save(train_dense_path, train_dense)
        np.save(test_dense_path, test_dense)

        return train_sparse, train_dense, test_sparse, test_dense


class NeuralFeaturePipeline:
    """
    Manages the tokenization of text for Transformer models.
    """

    def __init__(self, config):
        self.config = config
        self.model_name = config.get("transformer_model", "roberta-base")
        self.max_len = config.get("max_length", 80)

    def execute(self, train_text, test_text, load_cached_data=True):
        """
        Tokenizes train and test text and caches the resulting arrays.

        Args:
            train_text (pd.Series or list): Training text.
            test_text (pd.Series or list): Test text.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (train_encodings, test_encodings)
                   Each is a dict with 'input_ids' and 'attention_mask' (numpy arrays).
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        config_hash = generate_config_hash(self.config)

        # Define paths
        train_ids_path = os.path.join(CACHE_DIR, f"train_ids_{config_hash}.npy")
        train_mask_path = os.path.join(CACHE_DIR, f"train_mask_{config_hash}.npy")
        test_ids_path = os.path.join(CACHE_DIR, f"test_ids_{config_hash}.npy")
        test_mask_path = os.path.join(CACHE_DIR, f"test_mask_{config_hash}.npy")

        # Check cache
        if load_cached_data and all(
            os.path.exists(p)
            for p in [train_ids_path, train_mask_path, test_ids_path, test_mask_path]
        ):
            print(f"Loading neural features from cache: {CACHE_DIR}")
            train_ids = np.load(train_ids_path)
            train_mask = np.load(train_mask_path)
            test_ids = np.load(test_ids_path)
            test_mask = np.load(test_mask_path)

            return (
                {"input_ids": train_ids, "attention_mask": train_mask},
                {"input_ids": test_ids, "attention_mask": test_mask},
            )

        print(f"Tokenizing text with {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def tokenize_batch(texts):
            if isinstance(texts, pd.Series):
                texts = texts.tolist()
            return tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="np",
                return_attention_mask=True,
            )

        train_enc = tokenize_batch(train_text)
        test_enc = tokenize_batch(test_text)

        # Save to cache
        print(f"Saving neural features to {CACHE_DIR}...")
        np.save(train_ids_path, train_enc["input_ids"])
        np.save(train_mask_path, train_enc["attention_mask"])
        np.save(test_ids_path, test_enc["input_ids"])
        np.save(test_mask_path, test_enc["attention_mask"])

        return (
            {
                "input_ids": train_enc["input_ids"],
                "attention_mask": train_enc["attention_mask"],
            },
            {
                "input_ids": test_enc["input_ids"],
                "attention_mask": test_enc["attention_mask"],
            },
        )
