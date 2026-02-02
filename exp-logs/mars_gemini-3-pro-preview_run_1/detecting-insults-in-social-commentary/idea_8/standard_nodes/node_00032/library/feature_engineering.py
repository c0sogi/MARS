import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from library.config import Config
from library.utils import seed_everything


class StructuralFeatureGenerator:
    """
    Generates structural features from text data using TF-IDF and TruncatedSVD.
    Implements a caching mechanism to avoid redundant computation.
    Strictly fits transformations on training data only to prevent leakage.
    """

    def __init__(self):
        self.config = Config
        seed_everything(self.config.seed)

    def generate_features(self, load_cached_data: bool = True, debug: bool = False):
        """
        Main method to generate or load structural features.

        Args:
            load_cached_data (bool): If True, attempts to load features from disk.
            debug (bool): If True, runs on a subset of data and skips saving to production cache.

        Returns:
            tuple: (train_features, val_features, test_features) as numpy arrays.
        """
        # Determine paths
        train_path = self.config.train_struct_features_path
        val_path = self.config.val_struct_features_path
        test_path = self.config.test_struct_features_path

        # If debugging, we force re-computation and do not use the main cache files
        if debug:
            print("Debug mode active: Generating features on data subset...")
            return self._compute_features(debug=True)

        # Check if cache exists and loading is requested
        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            print("Loading structural features from cache...")
            try:
                train_features = np.load(train_path)
                val_features = np.load(val_path)
                test_features = np.load(test_path)
                return train_features, val_features, test_features
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing features...")

        # Compute from scratch
        print("Computing structural features from scratch...")
        train_features, val_features, test_features = self._compute_features(
            debug=False
        )

        # Save to cache
        print(f"Saving features to {self.config.working_dir}...")
        np.save(train_path, train_features)
        np.save(val_path, val_features)
        np.save(test_path, test_features)

        return train_features, val_features, test_features

    def _compute_features(self, debug: bool):
        """
        Internal method to perform the TF-IDF -> SVD -> Scaling pipeline.
        """
        # 1. Load Data
        df_train = pd.read_csv(self.config.train_path)
        df_val = pd.read_csv(self.config.val_path)
        df_test = pd.read_csv(self.config.test_path)

        # Handle Debugging
        if debug:
            df_train = df_train.head(100)
            df_val = df_val.head(50)
            df_test = df_test.head(50)

        # Fill NaNs
        train_text = df_train["Comment"].fillna("").astype(str).tolist()
        val_text = df_val["Comment"].fillna("").astype(str).tolist()
        test_text = df_test["Comment"].fillna("").astype(str).tolist()

        # 2. TF-IDF Vectorization
        # Word N-grams
        print("Fitting Word TF-IDF...")
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.config.tfidf_word_ngram_range,
            min_df=2,
            max_features=None,  # Let SVD handle dimensionality reduction
            sublinear_tf=True,
        )
        # Fit on TRAIN only
        train_word = word_vectorizer.fit_transform(train_text)
        val_word = word_vectorizer.transform(val_text)
        test_word = word_vectorizer.transform(test_text)

        # Char N-grams
        print("Fitting Char TF-IDF...")
        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.config.tfidf_char_ngram_range,
            min_df=2,
            max_features=None,
            sublinear_tf=True,
        )
        # Fit on TRAIN only
        train_char = char_vectorizer.fit_transform(train_text)
        val_char = char_vectorizer.transform(val_text)
        test_char = char_vectorizer.transform(test_text)

        # Stack features
        print("Stacking sparse matrices...")
        train_sparse = sparse.hstack([train_word, train_char])
        val_sparse = sparse.hstack([val_word, val_char])
        test_sparse = sparse.hstack([test_word, test_char])

        # 3. Truncated SVD
        print(f"Fitting TruncatedSVD (n_components={self.config.svd_output_dim})...")
        svd = TruncatedSVD(
            n_components=self.config.svd_output_dim,
            random_state=self.config.seed,
            n_iter=5,
        )
        # Fit on TRAIN only
        train_svd = svd.fit_transform(train_sparse)
        val_svd = svd.transform(val_sparse)
        test_svd = svd.transform(test_sparse)

        # 4. Normalization (Standard Scaling)
        # While the model may use LayerNorm, standard scaling helps SVD features
        # (which have decaying variance) be more amenable to neural network initialization.
        print("Applying Standard Scaling...")
        scaler = StandardScaler()
        # Fit on TRAIN only
        train_features = scaler.fit_transform(train_svd)
        val_features = scaler.transform(val_svd)
        test_features = scaler.transform(test_svd)

        # Ensure float32 for PyTorch compatibility
        train_features = train_features.astype(np.float32)
        val_features = val_features.astype(np.float32)
        test_features = test_features.astype(np.float32)

        return train_features, val_features, test_features
