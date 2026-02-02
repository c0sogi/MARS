import os
import re
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from library.config import Config
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles the generation of non-neural features for the essay scoring task.
    Includes Structural (linguistic), Lexical (word TF-IDF), and Morphological (char TF-IDF) features.
    """

    def __init__(self):
        seed_everything(Config.SEED)

        # Load datasets using metadata paths
        self.train_df = pd.read_csv(Config.TRAIN_PATH)
        self.val_df = pd.read_csv(Config.VAL_PATH)
        self.test_df = pd.read_csv(Config.TEST_PATH)

        # Handle Debug Mode
        if Config.DEBUG:
            self.train_df = self.train_df.head(50)
            self.val_df = self.val_df.head(50)
            self.test_df = self.test_df.head(50)

        # Preprocess text (minimal cleaning)
        self.train_df["clean_text"] = self.train_df["full_text"].apply(self._clean_text)
        self.val_df["clean_text"] = self.val_df["full_text"].apply(self._clean_text)
        self.test_df["clean_text"] = self.test_df["full_text"].apply(self._clean_text)

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def _clean_text(self, text):
        """Minimal whitespace normalization."""
        if pd.isna(text):
            return ""
        return " ".join(str(text).split())

    def _compute_structural_features(self, df, common_vocab=None):
        """
        Computes explicit linguistic features for a given dataframe.
        """

        # Helper functions for apply
        def get_stats(text):
            # Tokenization
            words = re.findall(r"\w+", text)
            sentences = re.split(r"[.!?]+", text)
            sentences = [s for s in sentences if s.strip()]  # Filter empty

            # Basic Counts
            char_count = len(text)
            word_count = len(words)
            sentence_count = len(sentences)

            # Derived metrics
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0

            # Vocabulary richness
            unique_words = set(w.lower() for w in words)
            unique_word_count = len(unique_words)
            unique_word_ratio = unique_word_count / word_count if word_count > 0 else 0

            # Punctuation
            punctuation_count = len(re.findall(r"[.,?!:;]", text))

            # Spelling/Rare word proxy
            # Count words not in common_vocab (if provided)
            spelling_error_count = 0
            if common_vocab is not None:
                spelling_error_count = sum(
                    1 for w in words if w.lower() not in common_vocab
                )

            return pd.Series(
                {
                    "word_count": word_count,
                    "char_count": char_count,
                    "sentence_count": sentence_count,
                    "avg_word_len": avg_word_len,
                    "avg_sentence_len": avg_sentence_len,
                    "unique_word_count": unique_word_count,
                    "unique_word_ratio": unique_word_ratio,
                    "punctuation_count": punctuation_count,
                    "spelling_error_count": spelling_error_count,
                }
            )

        # Apply feature extraction
        features = df["clean_text"].apply(get_stats)
        return features

    def get_structural_features(self, load_cached_data=True):
        """
        Generates or loads structural features for Train, Val, and Test sets.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            tuple: (train_features_df, val_features_df, test_features_df)
        """
        train_path = os.path.join(Config.CACHE_DIR, "structural_train.parquet")
        val_path = os.path.join(Config.CACHE_DIR, "structural_val.parquet")
        test_path = os.path.join(Config.CACHE_DIR, "structural_test.parquet")

        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            print("Loading structural features from cache...")
            train_feats = pd.read_parquet(train_path)
            val_feats = pd.read_parquet(val_path)
            test_feats = pd.read_parquet(test_path)
            return train_feats, val_feats, test_feats

        print("Computing structural features...")

        # Build common vocabulary from training data for spelling proxy
        # We consider words appearing at least 10 times as "common/correct"
        all_train_words = []
        for text in self.train_df["clean_text"]:
            all_train_words.extend(re.findall(r"\w+", text.lower()))

        word_counts = Counter(all_train_words)
        common_vocab = set(w for w, c in word_counts.items() if c >= 10)

        # Compute features
        train_feats = self._compute_structural_features(self.train_df, common_vocab)
        val_feats = self._compute_structural_features(self.val_df, common_vocab)
        test_feats = self._compute_structural_features(self.test_df, common_vocab)

        # Save to cache
        print("Saving structural features to cache...")
        train_feats.to_parquet(train_path)
        val_feats.to_parquet(val_path)
        test_feats.to_parquet(test_path)

        return train_feats, val_feats, test_feats

    def get_tfidf_features(self, kind="word", load_cached_data=True):
        """
        Generates or loads TF-IDF features.

        Args:
            kind (str): 'word' for Lexical branch, 'char' for Morphological branch.
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            tuple: (train_csr, val_csr, test_csr)
        """
        if kind not in ["word", "char"]:
            raise ValueError("kind must be 'word' or 'char'")

        train_path = os.path.join(Config.CACHE_DIR, f"tfidf_{kind}_train.npz")
        val_path = os.path.join(Config.CACHE_DIR, f"tfidf_{kind}_val.npz")
        test_path = os.path.join(Config.CACHE_DIR, f"tfidf_{kind}_test.npz")

        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            print(f"Loading {kind} TF-IDF features from cache...")
            train_csr = scipy.sparse.load_npz(train_path)
            val_csr = scipy.sparse.load_npz(val_path)
            test_csr = scipy.sparse.load_npz(test_path)
            return train_csr, val_csr, test_csr

        print(f"Computing {kind} TF-IDF features...")

        # Configure Vectorizer based on Config
        if kind == "word":
            vectorizer = TfidfVectorizer(
                ngram_range=Config.WORD_NGRAM_RANGE,
                min_df=Config.WORD_MIN_DF,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\w{1,}",
                stop_words="english",
            )
        else:  # char
            vectorizer = TfidfVectorizer(
                ngram_range=Config.CHAR_NGRAM_RANGE,
                min_df=Config.CHAR_MIN_DF,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="char",
            )

        # Fit on Train, Transform All
        train_csr = vectorizer.fit_transform(self.train_df["clean_text"])
        val_csr = vectorizer.transform(self.val_df["clean_text"])
        test_csr = vectorizer.transform(self.test_df["clean_text"])

        # Save to cache
        print(f"Saving {kind} TF-IDF features to cache...")
        scipy.sparse.save_npz(train_path, train_csr)
        scipy.sparse.save_npz(val_path, val_csr)
        scipy.sparse.save_npz(test_path, test_csr)

        return train_csr, val_csr, test_csr
