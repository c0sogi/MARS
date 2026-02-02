import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class TfidfEmbedder:
    """
    Wrapper around sklearn's TfidfVectorizer to handle text to vector conversion.
    """

    def __init__(self, max_features=100000, ngram_range=(1, 2), verbose=False):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.verbose = verbose
        # Initialize the vectorizer
        # sublinear_tf=True scales term frequency logarithmically (1+log(tf))
        # dtype=np.float32 reduces memory usage
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
            dtype=np.float32,
            token_pattern=r"(?u)\b\w\w+\b",
        )

    def fit(self, texts):
        """
        Fits the vectorizer to the provided texts.
        """
        if self.verbose:
            print(f"Fitting TfidfVectorizer on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        """
        Transforms the texts into TF-IDF vectors.
        """
        if self.verbose:
            print(f"Transforming {len(texts)} documents...")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """
        Fits and transforms the texts.
        """
        if self.verbose:
            print(f"Fitting and transforming {len(texts)} documents...")
        return self.vectorizer.fit_transform(texts)

    def save(self, path):
        """
        Saves the vectorizer object to disk.
        """
        if self.verbose:
            print(f"Saving TfidfEmbedder to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)

    def load(self, path):
        """
        Loads the vectorizer object from disk.
        """
        if self.verbose:
            print(f"Loading TfidfEmbedder from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.vectorizer = joblib.load(path)
        return self


def _save_sparse_matrix(matrix, path):
    """
    Helper to save a sparse matrix using numpy .npz format (avoiding pickle).
    """
    if not isinstance(matrix, sparse.csr_matrix):
        matrix = matrix.tocsr()

    np.savez(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,
    )


def _load_sparse_matrix(path):
    """
    Helper to load a sparse matrix from numpy .npz format.
    """
    loader = np.load(path)
    return sparse.csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
    )


def get_tfidf_features(
    df,
    split: str,
    embedder: TfidfEmbedder = None,
    load_cached_data: bool = True,
    cache_dir: str = "./working/idea_2",
):
    """
    Retrieves TF-IDF features for a given dataframe split.
    Handles caching of the resulting sparse matrix.

    Args:
        df (pd.DataFrame): Dataframe containing a 'text' column.
        split (str): The split name ('train', 'val', 'test').
        embedder (TfidfEmbedder): The embedder instance. Required if computing from scratch.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_dir (str): Directory to store cached files.

    Returns:
        scipy.sparse.csr_matrix: The TF-IDF feature matrix.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}_tfidf_features.npz")

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            print(f"Loading cached TF-IDF features for '{split}' from {cache_path}...")
            try:
                return _load_sparse_matrix(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding to re-process.")
        else:
            print(f"No cache found for '{split}' at {cache_path}.")

    # 2. Compute features
    if embedder is None:
        raise ValueError("Embedder instance must be provided to compute features.")

    print(f"Computing TF-IDF features for '{split}'...")
    texts = df["text"].fillna("").tolist()

    # For training set, we fit_transform. For others, we transform.
    # Note: This implies the embedder is modified in-place for 'train'.
    if split == "train":
        features = embedder.fit_transform(texts)
    else:
        features = embedder.transform(texts)

    # 3. Save to cache
    print(f"Saving processed features to {cache_path}...")
    _save_sparse_matrix(features, cache_path)

    return features
