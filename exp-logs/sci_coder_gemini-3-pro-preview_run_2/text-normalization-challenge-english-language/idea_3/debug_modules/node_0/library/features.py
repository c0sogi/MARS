import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from library.config import (
    TRAIN_FILE,
    TEST_FILE,
    TRAIN_VECTORS_PATH,
    TRAIN_LABELS_PATH,
    WORK_DIR,
    SEED,
    PLAIN_SAMPLE_RATIO,
    EMBEDDING_DIM,
)

# Path for the serialized embedder model
EMBEDDER_PATH = os.path.join(WORK_DIR, "subword_embedder.joblib")
TEST_VECTORS_PATH = os.path.join(WORK_DIR, "test_vectors.npy")


class SubwordEmbedder:
    """
    Approximates FastText-like subword embeddings using TfidfVectorizer on character n-grams
    followed by TruncatedSVD for dimensionality reduction.
    """

    def __init__(self, embedding_dim=100, seed=42):
        self.embedding_dim = embedding_dim
        self.seed = seed
        # Character n-grams (1-4) capture subword information (prefixes, suffixes, stems)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=2,  # Ignore extremely rare n-grams
            max_features=100000,  # Cap features to manage memory
            dtype=np.float32,
        )
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=seed)
        self.is_fitted = False

    def fit(self, texts):
        """
        Fits the TF-IDF vectorizer and SVD on the provided texts.
        """
        print("Fitting TfidfVectorizer...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        print(f"Fitting TruncatedSVD on shape {tfidf_matrix.shape}...")
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True
        return self

    def _get_embeddings(self, texts):
        """
        Helper to transform raw texts into dense vectors.
        """
        if not self.is_fitted:
            raise ValueError("SubwordEmbedder is not fitted yet.")

        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return dense

    def transform(self, df):
        """
        Generates context-aware embeddings for tokens in the dataframe.
        Input df must have 'sentence_id' and 'before' columns.

        Returns:
            np.ndarray: Matrix of shape (n_samples, embedding_dim * 2)
                        [target_vector, context_vector]
        """
        # 1. Efficient Context Extraction (Vectorized)
        # We shift the 'before' column to get prev and next tokens.
        # We then mask out shifts that cross sentence boundaries.
        print("Extracting context tokens...")

        # Ensure string type
        tokens = df["before"].astype(str).fillna("")
        sentence_ids = df["sentence_id"].values

        # Shift
        prev_tokens = tokens.shift(1).fillna("")
        next_tokens = tokens.shift(-1).fillna("")

        # Check boundaries
        # If sentence_id[i] != sentence_id[i-1], then prev_token is invalid (start of sent)
        # If sentence_id[i] != sentence_id[i+1], then next_token is invalid (end of sent)

        # Create boolean masks
        # Note: We compare with shifted sentence_ids.
        # For the first element, shift(1) is NaN, so inequality holds.
        same_as_prev = sentence_ids == np.roll(sentence_ids, 1)
        same_as_prev[0] = False  # First element has no prev in same sentence

        same_as_next = sentence_ids == np.roll(sentence_ids, -1)
        same_as_next[-1] = False  # Last element has no next in same sentence

        # Apply masks
        prev_tokens = np.where(same_as_prev, prev_tokens, "")
        next_tokens = np.where(same_as_next, next_tokens, "")

        # 2. Compute Embeddings
        print("Computing embeddings for target tokens...")
        target_vecs = self._get_embeddings(tokens)

        print("Computing embeddings for context tokens...")
        # We process unique tokens to save time?
        # Given the SVD is fast on sparse input, direct transform is usually fine for 1M-7M rows
        # if batching is handled by sklearn.
        prev_vecs = self._get_embeddings(prev_tokens)
        next_vecs = self._get_embeddings(next_tokens)

        # 3. Combine Context
        # Context vector is average of prev and next
        context_vecs = (prev_vecs + next_vecs) / 2.0

        # 4. Concatenate [Target, Context]
        print("Concatenating features...")
        features = np.hstack([target_vecs, context_vecs])

        return features.astype(np.float32)

    def save(self, path):
        joblib.dump(self, path)
        print(f"Embedder saved to {path}")

    @classmethod
    def load(cls, path):
        print(f"Loading embedder from {path}...")
        return joblib.load(path)


def load_or_create_train_features(load_cached_data=True):
    """
    Loads training features from cache or creates them from scratch.
    Returns:
        tuple: (train_vectors, train_labels, embedder)
    """
    # Check cache
    if (
        load_cached_data
        and os.path.exists(TRAIN_VECTORS_PATH)
        and os.path.exists(TRAIN_LABELS_PATH)
    ):
        print("Loading training features from cache...")
        vectors = np.load(TRAIN_VECTORS_PATH)
        labels = np.load(TRAIN_LABELS_PATH)

        # Try to load embedder if it exists, otherwise we might need to refit (but usually it should exist)
        if os.path.exists(EMBEDDER_PATH):
            embedder = SubwordEmbedder.load(EMBEDDER_PATH)
        else:
            print(
                "Warning: Cached vectors found but Embedder model missing. Re-fitting embedder not implemented here."
            )
            embedder = None

        return vectors, labels, embedder

    print("Generating training features from scratch...")

    # 1. Load Data
    print(f"Loading {TRAIN_FILE}...")
    df = pd.read_csv(
        TRAIN_FILE,
        keep_default_na=False,
        dtype={
            "sentence_id": int,
            "token_id": int,
            "class": str,
            "before": str,
            "after": str,
        },
    )

    # 2. Downsample PLAIN class
    print(f"Downsampling PLAIN class (Ratio: {PLAIN_SAMPLE_RATIO})...")
    df_plain = df[df["class"] == "PLAIN"]
    df_others = df[df["class"] != "PLAIN"]

    # Sample PLAIN
    df_plain_sampled = df_plain.sample(frac=PLAIN_SAMPLE_RATIO, random_state=SEED)

    # Combine and Shuffle
    df_train = pd.concat([df_others, df_plain_sampled], axis=0)
    df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"Training set size after downsampling: {len(df_train)}")

    # 3. Fit Embedder
    embedder = SubwordEmbedder(embedding_dim=EMBEDDING_DIM, seed=SEED)
    # Fit on the 'before' text of the downsampled training set
    # (or could fit on full set, but downsampled is likely sufficient and faster)
    embedder.fit(df_train["before"].values)

    # Save embedder
    embedder.save(EMBEDDER_PATH)

    # 4. Transform
    vectors = embedder.transform(df_train)
    labels = df_train["class"].values

    # 5. Save Cache
    print("Saving vectors and labels to cache...")
    np.save(TRAIN_VECTORS_PATH, vectors)
    np.save(TRAIN_LABELS_PATH, labels)

    return vectors, labels, embedder


def create_test_features(embedder, load_cached_data=True):
    """
    Generates features for the test set using the provided embedder.
    """
    if load_cached_data and os.path.exists(TEST_VECTORS_PATH):
        print("Loading test features from cache...")
        return np.load(TEST_VECTORS_PATH)

    print("Generating test features...")

    # Load Data
    df_test = pd.read_csv(
        TEST_FILE,
        keep_default_na=False,
        dtype={"sentence_id": int, "token_id": int, "before": str},
    )

    if embedder is None:
        if os.path.exists(EMBEDDER_PATH):
            embedder = SubwordEmbedder.load(EMBEDDER_PATH)
        else:
            raise ValueError(
                "Embedder not provided and not found on disk. Cannot process test set."
            )

    # Transform
    vectors = embedder.transform(df_test)

    # Cache
    print("Saving test vectors to cache...")
    np.save(TEST_VECTORS_PATH, vectors)

    return vectors
