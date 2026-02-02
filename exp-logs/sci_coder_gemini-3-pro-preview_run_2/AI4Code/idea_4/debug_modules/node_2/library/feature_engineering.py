import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from library.config import Config


class SparseFeaturePipeline:
    """
    Manages the TF-IDF Vectorization for the Sparse Stream (Ridge Regression).
    Operates on the Markdown content of the notebooks.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,  # Logarithmic TF scaling is beneficial for ranking
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
        )
        self.path = Config.RIDGE_MODEL_PATH.replace(
            "ridge_model.joblib", "tfidf_vectorizer.joblib"
        )
        # Note: Config defines TFIDF_VECTORIZER_PATH, let's use that if available
        if hasattr(Config, "TFIDF_VECTORIZER_PATH"):
            self.path = Config.TFIDF_VECTORIZER_PATH

    def fit(self, corpus):
        """
        Fits the vectorizer on the training corpus.
        Args:
            corpus (iterable): List or Series of markdown text strings.
        """
        print(f"Fitting Sparse TF-IDF Vectorizer on {len(corpus)} samples...")
        self.vectorizer.fit(corpus)
        return self

    def transform(self, corpus):
        """
        Transforms the corpus into a sparse matrix.
        Args:
            corpus (iterable): List or Series of markdown text strings.
        Returns:
            scipy.sparse.csr_matrix: The TF-IDF features.
        """
        print(f"Transforming {len(corpus)} samples to Sparse TF-IDF...")
        return self.vectorizer.transform(corpus)

    def fit_transform(self, corpus):
        print(f"Fitting and Transforming {len(corpus)} samples...")
        return self.vectorizer.fit_transform(corpus)

    def save(self):
        """Saves the fitted vectorizer to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        joblib.dump(self.vectorizer, self.path)
        print(f"Sparse Vectorizer saved to {self.path}")

    def load(self):
        """Loads the fitted vectorizer from disk."""
        if os.path.exists(self.path):
            self.vectorizer = joblib.load(self.path)
            print(f"Sparse Vectorizer loaded from {self.path}")
        else:
            raise FileNotFoundError(
                f"Vectorizer not found at {self.path}. Call fit() first."
            )
        return self


class ContextExtractor:
    """
    Manages the TF-IDF Vectorization for the Dense Stream Context.
    Operates on the Code content of the notebooks to extract high-signal keywords.
    """

    def __init__(self):
        # Configuration matches the logic in data_processing.py
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
            min_df=5,
        )
        self.path = Config.CODE_TFIDF_VECTORIZER_PATH
        self.top_k = Config.MAX_CODE_TOKENS_CONTEXT

    def fit(self, corpus):
        """
        Fits the vectorizer on the code corpus.
        Args:
            corpus (iterable): List of code strings (one per notebook).
        """
        print(f"Fitting Code Context Vectorizer on {len(corpus)} notebooks...")
        self.vectorizer.fit(corpus)
        return self

    def extract_context(self, corpus):
        """
        Extracts the top K keywords for each document in the corpus.
        Args:
            corpus (iterable): List of code strings.
        Returns:
            list: List of space-separated keyword strings.
        """
        print(f"Extracting context keywords for {len(corpus)} notebooks...")
        tfidf_matrix = self.vectorizer.transform(corpus)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        contexts = []
        # Process in chunks or iterate (tqdm not strictly required here per instructions but logic needed)
        # Using sparse matrix iteration
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i]
            _, col_indices = row.nonzero()

            if len(col_indices) == 0:
                contexts.append("")
                continue

            data = row.data
            # Sort by score descending
            sorted_indices = np.argsort(data)[::-1]
            top_k_indices = sorted_indices[: self.top_k]
            top_feat_indices = col_indices[top_k_indices]

            keywords = feature_names[top_feat_indices]
            contexts.append(" ".join(keywords))

        return contexts

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        joblib.dump(self.vectorizer, self.path)
        print(f"Code Vectorizer saved to {self.path}")

    def load(self):
        if os.path.exists(self.path):
            self.vectorizer = joblib.load(self.path)
            print(f"Code Vectorizer loaded from {self.path}")
        else:
            raise FileNotFoundError(f"Code Vectorizer not found at {self.path}.")
        return self


def create_sparse_features(df, split, load_cached_data=True):
    """
    Generates or loads the sparse TF-IDF feature matrix for the given dataframe.
    Implements strict caching logic using .npz files.

    Args:
        df (pd.DataFrame): Dataframe containing a 'source' column with markdown text.
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature matrix.
    """
    cache_filename = f"{split}_sparse_features.npz"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached sparse features from {cache_path}...")
        try:
            features = sparse.load_npz(cache_path)
            return features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Generating sparse features for {split}...")
    pipeline = SparseFeaturePipeline()

    corpus = df["source"].astype(str).fillna("")

    if split == "train":
        # Fit and transform
        features = pipeline.fit_transform(corpus)
        pipeline.save()
    else:
        # Load existing vectorizer and transform
        try:
            pipeline.load()
        except FileNotFoundError:
            if split == "val":
                # Fallback for validation if running without prior training (should not happen in pipeline)
                print(
                    "Warning: Train vectorizer not found during validation. Fitting on val (suboptimal)."
                )
                features = pipeline.fit_transform(corpus)
            else:
                raise RuntimeError("Cannot process test data: Vectorizer not found.")
        else:
            features = pipeline.transform(corpus)

    # 3. Save to Cache
    print(f"Saving sparse features to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sparse.save_npz(cache_path, features)

    return features
