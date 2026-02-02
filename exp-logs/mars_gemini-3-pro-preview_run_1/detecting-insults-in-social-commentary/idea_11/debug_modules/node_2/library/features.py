import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class SVDFeatureExtractor:
    """
    Handles the extraction of structural features using TF-IDF and SVD.
    Ensures no data leakage by fitting only on the training set.
    """

    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=Config.ngram_range_word,
            min_df=2,
            max_features=None,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=Config.ngram_range_char,
            min_df=2,
            max_features=None,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(
            n_components=Config.svd_components,
            random_state=Config.seed,
            algorithm="arpack",
        )

    def fit_transform_train_test(self, train_texts, val_texts, test_texts):
        """
        Fits transformers on train_texts, then transforms train, val, and test texts.

        Args:
            train_texts (list or pd.Series): Training text data.
            val_texts (list or pd.Series): Validation text data.
            test_texts (list or pd.Series): Test text data.

        Returns:
            tuple: (train_svd, val_svd, test_svd) as numpy arrays.
        """
        # 1. Fit and Transform Word N-grams (Fit on Train only)
        train_word = self.word_vectorizer.fit_transform(train_texts)
        val_word = self.word_vectorizer.transform(val_texts)
        test_word = self.word_vectorizer.transform(test_texts)

        # 2. Fit and Transform Char N-grams (Fit on Train only)
        train_char = self.char_vectorizer.fit_transform(train_texts)
        val_char = self.char_vectorizer.transform(val_texts)
        test_char = self.char_vectorizer.transform(test_texts)

        # 3. Concatenate Sparse Matrices
        train_sparse = scipy.sparse.hstack([train_word, train_char])
        val_sparse = scipy.sparse.hstack([val_word, val_char])
        test_sparse = scipy.sparse.hstack([test_word, test_char])

        # 4. Fit SVD on Train and Transform all
        # We fit SVD on the combined sparse matrix of the training set
        train_svd = self.svd.fit_transform(train_sparse)
        val_svd = self.svd.transform(val_sparse)
        test_svd = self.svd.transform(test_sparse)

        return (
            train_svd.astype(np.float32),
            val_svd.astype(np.float32),
            test_svd.astype(np.float32),
        )


def get_fold_features(
    fold_idx, train_texts, val_texts, test_texts, load_cached_data=True
):
    """
    Wrapper to generate or load SVD features for a specific fold.
    Implements caching mechanism using .npy files.

    Args:
        fold_idx (int): The current fold index.
        train_texts (list): Training texts for this fold.
        val_texts (list): Validation texts for this fold.
        test_texts (list): Test texts.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_svd, val_svd, test_svd)
    """
    # Define cache paths
    cache_dir = Config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, f"fold_{fold_idx}_train_svd.npy")
    val_path = os.path.join(cache_dir, f"fold_{fold_idx}_val_svd.npy")
    test_path = os.path.join(cache_dir, f"fold_{fold_idx}_test_svd.npy")

    # Check if files exist and loading is requested
    if load_cached_data:
        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            try:
                train_svd = np.load(train_path)
                val_svd = np.load(val_path)
                test_svd = np.load(test_path)
                return train_svd, val_svd, test_svd
            except Exception:
                # If load fails, proceed to compute
                pass

    # Compute features
    extractor = SVDFeatureExtractor()
    train_svd, val_svd, test_svd = extractor.fit_transform_train_test(
        train_texts, val_texts, test_texts
    )

    # Save to cache
    np.save(train_path, train_svd)
    np.save(val_path, val_svd)
    np.save(test_path, test_svd)

    return train_svd, val_svd, test_svd
