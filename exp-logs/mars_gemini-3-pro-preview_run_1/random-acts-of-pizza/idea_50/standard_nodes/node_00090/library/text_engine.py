import os
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy import sparse
import torch
from library.config import Config

# Attempt to ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass


class TextProcessor:
    def __init__(self):
        self.device = Config.DEVICE
        self.sbert_model_name = Config.SBERT_MODEL_NAME
        self.cache_dir = Config.CACHE_DIR

        # Models are initialized lazily
        self.sbert_model = None
        self.vader_analyzer = None

    def _load_sbert(self):
        """Lazily loads the SBERT model."""
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(
                self.sbert_model_name, device=self.device
            )

    def _load_vader(self):
        """Lazily loads the VADER analyzer."""
        if self.vader_analyzer is None:
            self.vader_analyzer = SentimentIntensityAnalyzer()

    def get_sbert_embeddings(self, texts, cache_name, load_cached_data=True):
        """
        Generates or loads SBERT embeddings for a list of texts.

        Args:
            texts (list or pd.Series): List of text strings to embed.
            cache_name (str): Unique identifier for the cache file (e.g. 'train_title').
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            np.ndarray: Embeddings of shape (N, embedding_dim).
        """
        cache_path = os.path.join(self.cache_dir, f"sbert_{cache_name}.npy")

        if load_cached_data and os.path.exists(cache_path):
            # Cite debug_lesson_7: Validate cache dimensions against current input
            cached_data = np.load(cache_path)
            if len(cached_data) == len(texts):
                print(f"Loading SBERT embeddings from cache: {cache_path}")
                return cached_data
            print(f"Cache dimension mismatch for {cache_name}. Regenerating...")

        print(f"Generating SBERT embeddings for {cache_name}...")
        self._load_sbert()

        # Handle NaN and ensure strings
        cleaned_texts = [str(t) if pd.notnull(t) else "" for t in texts]

        # Generate embeddings
        # normalize_embeddings=True is useful for cosine similarity features later
        embeddings = self.sbert_model.encode(
            cleaned_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Save to cache
        np.save(cache_path, embeddings)

        return embeddings

    def get_tfidf_features(
        self, train_texts, val_texts, test_texts, load_cached_data=True
    ):
        """
        Generates or loads TF-IDF features. Fits on train_texts, transforms others.

        Args:
            train_texts (list/Series): Training texts.
            val_texts (list/Series): Validation texts.
            test_texts (list/Series): Test texts.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            tuple: (train_tfidf, val_tfidf, test_tfidf) as sparse matrices.
        """
        train_cache = os.path.join(self.cache_dir, "tfidf_train.npz")
        val_cache = os.path.join(self.cache_dir, "tfidf_val.npz")
        test_cache = os.path.join(self.cache_dir, "tfidf_test.npz")

        if (
            load_cached_data
            and os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            # Cite debug_lesson_7: Validate cache dimensions
            train_tfidf = sparse.load_npz(train_cache)
            val_tfidf = sparse.load_npz(val_cache)
            test_tfidf = sparse.load_npz(test_cache)

            if (
                train_tfidf.shape[0] == len(train_texts)
                and val_tfidf.shape[0] == len(val_texts)
                and test_tfidf.shape[0] == len(test_texts)
            ):
                print("Loading TF-IDF features from cache...")
                return train_tfidf, val_tfidf, test_tfidf
            print("TF-IDF cache dimension mismatch. Regenerating...")

        print("Generating TF-IDF features...")

        # Clean inputs
        train_clean = [str(t) if pd.notnull(t) else "" for t in train_texts]
        val_clean = [str(t) if pd.notnull(t) else "" for t in val_texts]
        test_clean = [str(t) if pd.notnull(t) else "" for t in test_texts]

        vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )

        train_tfidf = vectorizer.fit_transform(train_clean)
        val_tfidf = vectorizer.transform(val_clean)
        test_tfidf = vectorizer.transform(test_clean)

        # Save to cache
        sparse.save_npz(train_cache, train_tfidf)
        sparse.save_npz(val_cache, val_tfidf)
        sparse.save_npz(test_cache, test_tfidf)

        return train_tfidf, val_tfidf, test_tfidf

    def get_vader_sentiment(self, texts, cache_name, load_cached_data=True):
        """
        Generates or loads VADER sentiment scores.

        Args:
            texts (list/Series): Texts to analyze.
            cache_name (str): Identifier for cache.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            np.ndarray: Shape (N, 4) containing [neg, neu, pos, compound].
        """
        cache_path = os.path.join(self.cache_dir, f"vader_{cache_name}.npy")

        if load_cached_data and os.path.exists(cache_path):
            # Cite debug_lesson_7: Validate cache dimensions against current input
            cached_data = np.load(cache_path)
            if len(cached_data) == len(texts):
                print(f"Loading VADER sentiment from cache: {cache_path}")
                return cached_data
            print(f"Cache dimension mismatch for {cache_name}. Regenerating...")

        print(f"Generating VADER sentiment for {cache_name}...")
        self._load_vader()

        cleaned_texts = [str(t) if pd.notnull(t) else "" for t in texts]

        results = []
        for text in cleaned_texts:
            scores = self.vader_analyzer.polarity_scores(text)
            results.append(
                [scores["neg"], scores["neu"], scores["pos"], scores["compound"]]
            )

        results_array = np.array(results, dtype=np.float32)

        np.save(cache_path, results_array)

        return results_array
