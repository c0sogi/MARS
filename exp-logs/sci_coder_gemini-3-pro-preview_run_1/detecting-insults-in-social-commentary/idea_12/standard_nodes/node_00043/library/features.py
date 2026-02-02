import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from library.configuration import Config
from library.utilities import set_seed


class SVDFeatureExtractor:
    """
    Handles the generation of structural features using TF-IDF and Truncated SVD.
    Ensures strict separation of fitting (Train) and transforming (Val/Test) to prevent data leakage.
    """

    def __init__(self):
        self.config = Config
        set_seed(self.config.seed)

        # Define cache filenames. Append '_debug' if in debug mode to avoid overwriting full features.
        suffix = "_debug" if self.config.debug else ""
        self.train_cache_path = os.path.join(
            self.config.cache_dir, f"train_svd{suffix}.npy"
        )
        self.val_cache_path = os.path.join(
            self.config.cache_dir, f"val_svd{suffix}.npy"
        )
        self.test_cache_path = os.path.join(
            self.config.cache_dir, f"test_svd{suffix}.npy"
        )

    def process(self, load_cached_data=True):
        """
        Main execution method.
        1. Checks for cached data.
        2. If not found, loads metadata.
        3. Computes TF-IDF (Word + Char).
        4. Computes SVD.
        5. Caches and returns data.

        Args:
            load_cached_data (bool): If True, attempts to load from disk first.

        Returns:
            tuple: (train_svd, val_svd, test_svd) as numpy arrays.
        """
        # 1. Check Cache
        if load_cached_data:
            if (
                os.path.exists(self.train_cache_path)
                and os.path.exists(self.val_cache_path)
                and os.path.exists(self.test_cache_path)
            ):
                print(f"Loading cached SVD features from {self.config.cache_dir}...")
                train_svd = np.load(self.train_cache_path)
                val_svd = np.load(self.val_cache_path)
                test_svd = np.load(self.test_cache_path)
                return train_svd, val_svd, test_svd
            else:
                print("Cache miss. Computing features from scratch...")
        else:
            print("Ignoring cache. Computing features from scratch...")

        # 2. Load Data
        print("Loading metadata CSVs...")
        try:
            train_df = pd.read_csv(self.config.train_path)
            val_df = pd.read_csv(self.config.val_path)
            test_df = pd.read_csv(self.config.test_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find metadata files. Ensure metadata generation was successful. {e}"
            )

        # Handle Debug Slicing
        if self.config.debug:
            print(
                f"Debug Mode: Slicing datasets to {self.config.debug_sample_size} samples."
            )
            train_df = train_df.iloc[: self.config.debug_sample_size]
            val_df = val_df.iloc[: self.config.debug_sample_size]
            test_df = test_df.iloc[: self.config.debug_sample_size]

        # Preprocessing: Fill NaNs
        train_text = train_df["Comment"].fillna("").astype(str).tolist()
        val_text = val_df["Comment"].fillna("").astype(str).tolist()
        test_text = test_df["Comment"].fillna("").astype(str).tolist()

        # 3. TF-IDF Vectorization
        print("Fitting TF-IDF Vectorizers...")

        # Word N-grams
        word_vectorizer = TfidfVectorizer(
            ngram_range=self.config.tfidf_word_ngram_range,
            analyzer="word",
            min_df=2,
            sublinear_tf=True,
        )
        # Strictly fit on TRAIN only
        train_word = word_vectorizer.fit_transform(train_text)
        val_word = word_vectorizer.transform(val_text)
        test_word = word_vectorizer.transform(test_text)

        # Char N-grams
        char_vectorizer = TfidfVectorizer(
            ngram_range=self.config.tfidf_char_ngram_range,
            analyzer="char",
            min_df=2,
            sublinear_tf=True,
        )
        # Strictly fit on TRAIN only
        train_char = char_vectorizer.fit_transform(train_text)
        val_char = char_vectorizer.transform(val_text)
        test_char = char_vectorizer.transform(test_text)

        # Stack Matrices
        print("Stacking sparse matrices...")
        train_feats = sparse.hstack([train_word, train_char])
        val_feats = sparse.hstack([val_word, val_char])
        test_feats = sparse.hstack([test_word, test_char])

        # 4. Truncated SVD
        n_components = self.config.svd_components

        # Safety check for Debug mode or small datasets
        # n_components must be <= n_samples for SVD to run meaningfully without error in some implementations
        if train_feats.shape[0] < n_components:
            print(
                f"Warning: Training samples ({train_feats.shape[0]}) < SVD components ({n_components}). Adjusting components."
            )
            n_components = max(1, train_feats.shape[0] - 1)

        print(f"Fitting TruncatedSVD with {n_components} components...")
        svd = TruncatedSVD(n_components=n_components, random_state=self.config.seed)

        # Fit on TRAIN only
        train_svd = svd.fit_transform(train_feats)
        val_svd = svd.transform(val_feats)
        test_svd = svd.transform(test_feats)

        # Pad features if components were reduced (to maintain consistency with model input size)
        if train_svd.shape[1] < self.config.svd_components:
            pad_width = self.config.svd_components - train_svd.shape[1]
            print(
                f"Padding SVD features with {pad_width} zeros to match configuration..."
            )
            train_svd = np.pad(train_svd, ((0, 0), (0, pad_width)), "constant")
            val_svd = np.pad(val_svd, ((0, 0), (0, pad_width)), "constant")
            test_svd = np.pad(test_svd, ((0, 0), (0, pad_width)), "constant")

        # 5. Cache Results
        print("Saving features to cache...")
        os.makedirs(self.config.cache_dir, exist_ok=True)
        np.save(self.train_cache_path, train_svd)
        np.save(self.val_cache_path, val_svd)
        np.save(self.test_cache_path, test_svd)

        return train_svd, val_svd, test_svd
