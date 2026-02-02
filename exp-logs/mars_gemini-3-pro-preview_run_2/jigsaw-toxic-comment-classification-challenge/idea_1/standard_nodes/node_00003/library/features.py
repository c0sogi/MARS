import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config
from library.utils import setup_logger


class FeatureExtractor:
    """
    Handles feature extraction using TF-IDF on word and character n-grams.
    Implements a caching mechanism using .npy files to store sparse matrices.
    """

    def __init__(self):
        self.logger = setup_logger("feature_extractor")
        self.cache_dir = Config.CACHE_DIR

    def _save_sparse(self, matrix, name):
        """
        Saves a sparse CSR matrix to a directory using numpy .npy files for components.
        This avoids using pickle and adheres to the requirement of using .npy format.

        Args:
            matrix (scipy.sparse.csr_matrix): The matrix to save.
            name (str): The name of the dataset (e.g., 'X_train'), used as the folder name.
        """
        matrix_dir = os.path.join(self.cache_dir, name)
        os.makedirs(matrix_dir, exist_ok=True)

        # Save components of the CSR matrix
        np.save(os.path.join(matrix_dir, "data.npy"), matrix.data)
        np.save(os.path.join(matrix_dir, "indices.npy"), matrix.indices)
        np.save(os.path.join(matrix_dir, "indptr.npy"), matrix.indptr)
        np.save(os.path.join(matrix_dir, "shape.npy"), np.array(matrix.shape))

    def _load_sparse(self, name):
        """
        Loads a sparse CSR matrix from a directory of .npy component files.

        Args:
            name (str): The name of the dataset (e.g., 'X_train').

        Returns:
            scipy.sparse.csr_matrix: The reconstructed matrix.
        """
        matrix_dir = os.path.join(self.cache_dir, name)

        # Check if directory and files exist
        if not os.path.exists(matrix_dir):
            raise FileNotFoundError(f"Cache directory {matrix_dir} not found.")

        required_files = ["data.npy", "indices.npy", "indptr.npy", "shape.npy"]
        for f in required_files:
            if not os.path.exists(os.path.join(matrix_dir, f)):
                raise FileNotFoundError(f"Missing {f} in {matrix_dir}")

        # Load components
        data = np.load(os.path.join(matrix_dir, "data.npy"))
        indices = np.load(os.path.join(matrix_dir, "indices.npy"))
        indptr = np.load(os.path.join(matrix_dir, "indptr.npy"))
        shape = np.load(os.path.join(matrix_dir, "shape.npy"))

        # Reconstruct matrix
        return sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))

    def extract_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates TF-IDF features for train, val, and test sets.
        Combines Word N-grams and Character N-grams.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (X_train, X_val, X_test) as scipy.sparse.csr_matrix objects.
        """
        self.logger.info("Starting Feature Extraction...")

        # 1. Attempt to load from cache
        if load_cached_data:
            try:
                self.logger.info(f"Checking cache in {self.cache_dir}...")
                X_train = self._load_sparse("X_train")
                X_val = self._load_sparse("X_val")
                X_test = self._load_sparse("X_test")

                # Basic validation to ensure cache matches current input size
                if (
                    X_train.shape[0] == len(train_df)
                    and X_val.shape[0] == len(val_df)
                    and X_test.shape[0] == len(test_df)
                ):

                    self.logger.info("Successfully loaded features from cache.")
                    return X_train, X_val, X_test
                else:
                    self.logger.warning(
                        "Cached features dimensions do not match current data. Recomputing..."
                    )
            except (FileNotFoundError, IOError, ValueError) as e:
                self.logger.info(
                    f"Cache miss or load error ({str(e)}). Computing features from scratch..."
                )

        # 2. Compute Features
        self.logger.info("Initializing TF-IDF Vectorizers...")

        # Word-level TF-IDF
        word_vectorizer = TfidfVectorizer(
            ngram_range=Config.WORD_NGRAM_RANGE,
            max_features=Config.WORD_MAX_FEATURES,
            min_df=Config.WORD_MIN_DF,
            analyzer="word",
            token_pattern=r"\w{1,}",  # Capture single letter words which can be relevant in chats
            strip_accents="unicode",
            sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
        )

        # Character-level TF-IDF
        char_vectorizer = TfidfVectorizer(
            ngram_range=Config.CHAR_NGRAM_RANGE,
            max_features=Config.CHAR_MAX_FEATURES,
            min_df=Config.CHAR_MIN_DF,
            analyzer="char",
            strip_accents="unicode",
            sublinear_tf=True,
        )

        # Extract text columns
        train_text = train_df["comment_text"]
        val_text = val_df["comment_text"]
        test_text = test_df["comment_text"]

        # Fit and Transform
        # Note: We fit only on training data to prevent data leakage
        self.logger.info("Fitting and transforming Word features...")
        word_vectorizer.fit(train_text)
        train_word = word_vectorizer.transform(train_text)
        val_word = word_vectorizer.transform(val_text)
        test_word = word_vectorizer.transform(test_text)

        self.logger.info("Fitting and transforming Character features...")
        char_vectorizer.fit(train_text)
        train_char = char_vectorizer.transform(train_text)
        val_char = char_vectorizer.transform(val_text)
        test_char = char_vectorizer.transform(test_text)

        # Concatenate features
        self.logger.info("Concatenating Word and Character features...")
        X_train = sparse.hstack([train_word, train_char], format="csr")
        X_val = sparse.hstack([val_word, val_char], format="csr")
        X_test = sparse.hstack([test_word, test_char], format="csr")

        # Clean up intermediate memory
        del train_word, val_word, test_word
        del train_char, val_char, test_char
        del word_vectorizer, char_vectorizer

        # 3. Save to Cache
        self.logger.info("Saving computed features to cache...")
        try:
            self._save_sparse(X_train, "X_train")
            self._save_sparse(X_val, "X_val")
            self._save_sparse(X_test, "X_test")
            self.logger.info("Features saved successfully.")
        except Exception as e:
            self.logger.warning(f"Failed to save features to cache: {e}")

        self.logger.info(f"Feature Extraction Complete.")
        self.logger.info(f"Train Feature Shape: {X_train.shape}")

        return X_train, X_val, X_test
