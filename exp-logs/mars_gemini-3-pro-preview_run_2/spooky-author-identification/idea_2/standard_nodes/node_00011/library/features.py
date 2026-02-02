import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from library.config import Config


class HybridFeatureGenerator:
    """
    Generates features for Author Identification using Hybrid N-Grams and SVD.
    Manages caching of processed features to optimize runtime.
    """

    def __init__(self):
        """
        Initialize feature extractors based on Configuration.
        """
        # Word-level TF-IDF Vectorizer
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=Config.WORD_NGRAM_RANGE,
            min_df=Config.MIN_DF,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",  # Default, but explicit ensures consistency
        )

        # Character-level TF-IDF Vectorizer
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=Config.CHAR_NGRAM_RANGE, min_df=Config.MIN_DF, analyzer="char"
        )

        # Dimensionality Reduction for Dense Features
        self.svd = TruncatedSVD(
            n_components=Config.SVD_N_COMPONENTS, random_state=Config.SVD_RANDOM_STATE
        )

        # Target Encoder
        self.label_encoder = LabelEncoder()

    def _get_paths(self, debug):
        """
        Generates file paths for caching based on the debug state and Config hash.
        """
        suffix = "_debug" if debug else ""

        # We append the extension manually after getting the hashed path prefix
        paths = {
            "train_sparse": Config.get_cache_path(f"train_feat_sparse{suffix}")
            + ".npz",
            "val_sparse": Config.get_cache_path(f"val_feat_sparse{suffix}") + ".npz",
            "test_sparse": Config.get_cache_path(f"test_feat_sparse{suffix}") + ".npz",
            "train_dense": Config.get_cache_path(f"train_feat_dense{suffix}") + ".npy",
            "val_dense": Config.get_cache_path(f"val_feat_dense{suffix}") + ".npy",
            "test_dense": Config.get_cache_path(f"test_feat_dense{suffix}") + ".npy",
            "train_labels": Config.get_cache_path(f"train_labels{suffix}") + ".npy",
            "val_labels": Config.get_cache_path(f"val_labels{suffix}") + ".npy",
            "label_classes": Config.get_cache_path(f"label_classes{suffix}") + ".npy",
        }
        return paths

    def process(self, train_df, val_df, test_df, load_cached_data=True, debug=False):
        """
        Main processing pipeline:
        1. Checks cache.
        2. If miss, computes TF-IDF (Word+Char) and SVD.
        3. Encodes labels.
        4. Saves to cache.
        5. Returns dictionary of features.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): Whether running in debug mode (affects cache filenames).

        Returns:
            dict: Dictionary containing sparse matrices, dense arrays, and label arrays.
        """
        paths = self._get_paths(debug)

        # Check if all cache files exist
        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_exist:
            print(f"Loading features from cache (Debug={debug})...")
            try:
                # Load Sparse Matrices
                X_train_sparse = scipy.sparse.load_npz(paths["train_sparse"])
                X_val_sparse = scipy.sparse.load_npz(paths["val_sparse"])
                X_test_sparse = scipy.sparse.load_npz(paths["test_sparse"])

                # Load Dense Arrays
                X_train_dense = np.load(paths["train_dense"])
                X_val_dense = np.load(paths["val_dense"])
                X_test_dense = np.load(paths["test_dense"])

                # Load Labels
                y_train = np.load(paths["train_labels"])
                y_val = np.load(paths["val_labels"])

                # Restore LabelEncoder classes
                self.label_encoder.classes_ = np.load(
                    paths["label_classes"], allow_pickle=True
                )

                return {
                    "train_sparse": X_train_sparse,
                    "val_sparse": X_val_sparse,
                    "test_sparse": X_test_sparse,
                    "train_dense": X_train_dense,
                    "val_dense": X_val_dense,
                    "test_dense": X_test_dense,
                    "y_train": y_train,
                    "y_val": y_val,
                    "label_classes": self.label_encoder.classes_,
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Computing features from scratch (Debug={debug})...")

        # Ensure text is string and handle NaNs
        train_text = train_df["text"].fillna("").astype(str)
        val_text = val_df["text"].fillna("").astype(str)
        test_text = test_df["text"].fillna("").astype(str)

        # 1. Word N-Grams
        print("Fitting Word Vectorizer...")
        X_train_word = self.word_vectorizer.fit_transform(train_text)
        X_val_word = self.word_vectorizer.transform(val_text)
        X_test_word = self.word_vectorizer.transform(test_text)

        # 2. Char N-Grams
        print("Fitting Char Vectorizer...")
        X_train_char = self.char_vectorizer.fit_transform(train_text)
        X_val_char = self.char_vectorizer.transform(val_text)
        X_test_char = self.char_vectorizer.transform(test_text)

        # 3. Concatenate (Sparse)
        print("Concatenating Sparse Features...")
        X_train_sparse = scipy.sparse.hstack([X_train_word, X_train_char])
        X_val_sparse = scipy.sparse.hstack([X_val_word, X_val_char])
        X_test_sparse = scipy.sparse.hstack([X_test_word, X_test_char])

        # 4. SVD (Dense)
        print(f"Fitting SVD (n_components={Config.SVD_N_COMPONENTS})...")
        X_train_dense = self.svd.fit_transform(X_train_sparse)
        X_val_dense = self.svd.transform(X_val_sparse)
        X_test_dense = self.svd.transform(X_test_sparse)

        # 5. Encode Targets
        print("Encoding Targets...")
        y_train = self.label_encoder.fit_transform(train_df["author"])
        y_val = self.label_encoder.transform(val_df["author"])

        # 6. Save to Cache
        print("Saving features to cache...")
        # Ensure directory exists (Config.WORKING_DIR is created in Config, but good practice)
        os.makedirs(os.path.dirname(paths["train_sparse"]), exist_ok=True)

        scipy.sparse.save_npz(paths["train_sparse"], X_train_sparse)
        scipy.sparse.save_npz(paths["val_sparse"], X_val_sparse)
        scipy.sparse.save_npz(paths["test_sparse"], X_test_sparse)

        np.save(paths["train_dense"], X_train_dense)
        np.save(paths["val_dense"], X_val_dense)
        np.save(paths["test_dense"], X_test_dense)

        np.save(paths["train_labels"], y_train)
        np.save(paths["val_labels"], y_val)
        np.save(paths["label_classes"], self.label_encoder.classes_)

        return {
            "train_sparse": X_train_sparse,
            "val_sparse": X_val_sparse,
            "test_sparse": X_test_sparse,
            "train_dense": X_train_dense,
            "val_dense": X_val_dense,
            "test_dense": X_test_dense,
            "y_train": y_train,
            "y_val": y_val,
            "label_classes": self.label_encoder.classes_,
        }
