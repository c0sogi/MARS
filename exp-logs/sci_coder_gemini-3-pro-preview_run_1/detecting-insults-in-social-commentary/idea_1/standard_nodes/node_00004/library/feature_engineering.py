import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz, load_npz, csr_matrix


class DualTfidfVectorizer:
    """
    A vectorizer that combines word-level, character-level TF-IDF features, and meta-features.
    Cite solution_lesson_node_00002: Explicitly includes Character N-grams to handle noisy text.
    """

    def __init__(self):
        # Word-level vectorizer: Expanded to Trigrams
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=None,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            sublinear_tf=True,
            stop_words="english",
        )

        # Character-level vectorizer: Expanded range (1-5) and lower min_df
        # Captures punctuation (unigrams) and longer subword patterns.
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(1, 5),
            min_df=2,
            max_features=None,
            strip_accents="unicode",
            analyzer="char",
            sublinear_tf=True,
        )

    def _get_meta_features(self, X):
        """
        Computes structural meta-features (Caps ratio, punctuation usage).
        """
        meta = []
        for text in X:
            t = str(text)
            length = len(t)
            if length == 0:
                meta.append([0.0, 0.0, 0.0, 0.0])
                continue

            # Features normalized by length
            caps = sum(1 for c in t if c.isupper()) / length
            exclams = t.count("!") / length
            questions = t.count("?") / length
            # Ratio of non-alphanumeric chars (excluding spaces)
            puncts = sum(1 for c in t if not c.isalnum() and not c.isspace()) / length

            meta.append([caps, exclams, questions, puncts])

        return csr_matrix(meta)

    def fit(self, X, y=None):
        """
        Fits both word and character vectorizers to the input text.
        """
        self.word_vectorizer.fit(X)
        self.char_vectorizer.fit(X)
        return self

    def transform(self, X):
        """
        Transforms input text into a combined sparse feature matrix including meta-features.
        """
        word_features = self.word_vectorizer.transform(X)
        char_features = self.char_vectorizer.transform(X)
        meta_features = self._get_meta_features(X)

        # Concatenate sparse matrices horizontally
        return hstack([word_features, char_features, meta_features])

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data in a single step.
        """
        self.fit(X)
        return self.transform(X)


def extract_features(X_train, X_val, X_test, load_cached_data=True):
    """
    Extracts features for train, validation, and test sets using DualTfidfVectorizer.
    Implements deterministic caching using scipy.sparse.save_npz (based on .npy).

    Args:
        X_train (array-like): Training text data.
        X_val (array-like): Validation text data.
        X_test (array-like): Test text data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_feats, X_val_feats, X_test_feats) as sparse matrices.
    """
    cache_dir = "./working/idea_1"
    os.makedirs(cache_dir, exist_ok=True)

    # Version 2 of cache for new features
    train_path = os.path.join(cache_dir, "features_train_v2.npz")
    val_path = os.path.join(cache_dir, "features_val_v2.npz")
    test_path = os.path.join(cache_dir, "features_test_v2.npz")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    ):
        try:
            print("Loading features from cache...")
            X_train_feats = load_npz(train_path)
            X_val_feats = load_npz(val_path)
            X_test_feats = load_npz(test_path)

            if (
                X_train_feats.shape[0] != len(X_train)
                or X_val_feats.shape[0] != len(X_val)
                or X_test_feats.shape[0] != len(X_test)
            ):
                raise ValueError("Cached features dimension mismatch")

            return X_train_feats, X_val_feats, X_test_feats
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing features from scratch...")

    # Initialize and fit vectorizer on training data only
    vectorizer = DualTfidfVectorizer()
    vectorizer.fit(X_train)

    # Transform all splits
    print("Transforming training data...")
    X_train_feats = vectorizer.transform(X_train)

    print("Transforming validation data...")
    X_val_feats = vectorizer.transform(X_val)

    print("Transforming test data...")
    X_test_feats = vectorizer.transform(X_test)

    # 3. Save to cache
    print("Saving features to cache...")
    try:
        save_npz(train_path, X_train_feats)
        save_npz(val_path, X_val_feats)
        save_npz(test_path, X_test_feats)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return X_train_feats, X_val_feats, X_test_feats
