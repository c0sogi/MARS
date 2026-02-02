import os
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config
from library.utils import get_logger


class StructuralFeatureExtractor:
    """
    Extracts structural features from text using Multi-Granularity TF-IDF
    (Word N-grams 1-2, Char N-grams 3-5) and compresses them using TruncatedSVD.
    """

    def __init__(self):
        self.logger = get_logger()
        self.svd_dim = Config.svd_dim

        # Word N-grams (1-2)
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            use_idf=True,
        )

        # Character N-grams (3-5)
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            use_idf=True,
        )

        # Dimensionality Reduction
        self.svd = TruncatedSVD(
            n_components=self.svd_dim,
            random_state=Config.seed,
            algorithm="randomized",
            n_iter=5,
        )

        self.is_fitted = False

    def fit(self, texts):
        """
        Fits the TF-IDF vectorizers and SVD on the provided texts.
        Args:
            texts (iterable): List or Series of text strings.
        """
        self.logger.info("Fitting TF-IDF vectorizers...")
        # Fill NaNs
        texts = pd.Series(texts).fillna("").astype(str)

        # Fit vectorizers
        word_feats = self.word_vectorizer.fit_transform(texts)
        char_feats = self.char_vectorizer.fit_transform(texts)

        self.logger.info(f"Word vocab size: {len(self.word_vectorizer.vocabulary_)}")
        self.logger.info(f"Char vocab size: {len(self.char_vectorizer.vocabulary_)}")

        # Stack features
        combined_feats = scipy.sparse.hstack([word_feats, char_feats])

        self.logger.info("Fitting TruncatedSVD...")
        self.svd.fit(combined_feats)

        self.is_fitted = True
        self.logger.info(
            f"Explained variance ratio sum: {self.svd.explained_variance_ratio_.sum():.4f}"
        )

    def transform(self, texts):
        """
        Transforms texts into dense SVD features.
        Args:
            texts (iterable): List or Series of text strings.
        Returns:
            np.ndarray: Dense feature matrix of shape (n_samples, svd_dim).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Feature extractor must be fitted before calling transform."
            )

        # Fill NaNs
        texts = pd.Series(texts).fillna("").astype(str)

        # Transform using vectorizers
        word_feats = self.word_vectorizer.transform(texts)
        char_feats = self.char_vectorizer.transform(texts)

        # Stack
        combined_feats = scipy.sparse.hstack([word_feats, char_feats])

        # Reduce dimensionality
        dense_feats = self.svd.transform(combined_feats)

        return dense_feats.astype(np.float32)


def process_and_cache(load_cached_data=True):
    """
    Main function to generate or load structural features for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (train_features, val_features, test_features) as numpy arrays.
    """
    logger = get_logger()

    # Define filenames based on debug mode
    suffix = "_debug" if Config.debug else ""
    train_file = os.path.join(Config.working_dir, f"train_struct_features{suffix}.npy")
    val_file = os.path.join(Config.working_dir, f"val_struct_features{suffix}.npy")
    test_file = os.path.join(Config.working_dir, f"test_struct_features{suffix}.npy")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Check if files exist
    files_exist = (
        os.path.exists(train_file)
        and os.path.exists(val_file)
        and os.path.exists(test_file)
    )

    if load_cached_data and files_exist:
        logger.info(f"Loading cached structural features from {Config.working_dir}...")
        try:
            train_feats = np.load(train_file)
            val_feats = np.load(val_file)
            test_feats = np.load(test_file)
            logger.info("Successfully loaded cached features.")
            return train_feats, val_feats, test_feats
        except Exception as e:
            logger.warning(f"Failed to load cached files: {e}. Recomputing...")

    logger.info("Computing structural features from scratch...")

    # Load Metadata
    train_df = pd.read_csv(Config.train_meta_path)
    val_df = pd.read_csv(Config.val_meta_path)
    test_df = pd.read_csv(Config.test_meta_path)

    # Handle Debug Mode
    if Config.debug:
        logger.info(
            f"Debug mode enabled. Subsampling to {Config.debug_subset_size} rows."
        )
        train_df = train_df.iloc[: Config.debug_subset_size]
        val_df = val_df.iloc[: Config.debug_subset_size]
        test_df = test_df.iloc[: Config.debug_subset_size]

    # Initialize and Fit Extractor
    extractor = StructuralFeatureExtractor()

    # Fit only on Training Data to prevent leakage
    logger.info("Fitting extractor on training data...")
    extractor.fit(train_df["Comment"])

    # Transform all splits
    logger.info("Transforming training data...")
    train_feats = extractor.transform(train_df["Comment"])

    logger.info("Transforming validation data...")
    val_feats = extractor.transform(val_df["Comment"])

    logger.info("Transforming test data...")
    test_feats = extractor.transform(test_df["Comment"])

    # Save to cache
    logger.info(f"Saving features to {Config.working_dir}...")
    np.save(train_file, train_feats)
    np.save(val_file, val_feats)
    np.save(test_file, test_feats)

    logger.info("Feature extraction complete.")
    return train_feats, val_feats, test_feats
