import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import Timer, set_seed


class FeatureGenerator:
    """
    FeatureGenerator handles the creation of specific feature sets for the
    High-Fidelity Hept-View Stacking Ensemble. It implements strict caching
    and hygienic feature generation for Lexical, Behavioral, Semantic, and
    Metadata modalities.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Initialize the FeatureGenerator with processed DataFrames.

        Args:
            train_df (pd.DataFrame): The union training dataset (train + val).
            test_df (pd.DataFrame): The test dataset.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.cache_dir = Config.WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set seed for any stochastic operations within feature generation
        set_seed(Config.SEED)

    def get_lexical_features(self, load_cached_data: bool = True):
        """
        Generates Sparse Lexical Features using TF-IDF on the combined Title + Body text.

        Key Configuration:
            - token_pattern=r"\w{1,}" to capture single characters (e.g., "I", "$").
            - min_df=2 to preserve the long tail of the vocabulary.

        Returns:
            tuple: (X_train_lexical, X_test_lexical) as scipy sparse matrices.
        """
        train_path = os.path.join(self.cache_dir, "X_train_lexical.npz")
        test_path = os.path.join(self.cache_dir, "X_test_lexical.npz")

        # 1. Attempt Load from Cache
        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Lexical features...")
            with Timer("Load Lexical Cache"):
                X_train = scipy.sparse.load_npz(train_path)
                X_test = scipy.sparse.load_npz(test_path)
                return X_train, X_test

        # 2. Generate from Scratch
        print("Generating Lexical features from scratch...")
        with Timer("Lexical Feature Generation"):
            # Initialize Vectorizer with High-Fidelity params
            vectorizer = TfidfVectorizer(**Config.LEXICAL_VECTORIZER_PARAMS)

            # Fit on Train, Transform Train & Test
            # Note: text_combined is generated in DataFactory
            X_train = vectorizer.fit_transform(self.train_df["text_combined"])
            X_test = vectorizer.transform(self.test_df["text_combined"])

            # 3. Save to Cache
            scipy.sparse.save_npz(train_path, X_train)
            scipy.sparse.save_npz(test_path, X_test)

        return X_train, X_test

    def get_behavioral_features(self, load_cached_data: bool = True):
        """
        Generates Sparse Behavioral Features using TF-IDF on Subreddit history.

        Key Configuration:
            - max_features=1000 (Bag-of-Concepts approach).

        Returns:
            tuple: (X_train_community, X_test_community) as scipy sparse matrices.
        """
        train_path = os.path.join(self.cache_dir, "X_train_community.npz")
        test_path = os.path.join(self.cache_dir, "X_test_community.npz")

        # 1. Attempt Load from Cache
        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Behavioral features...")
            with Timer("Load Behavioral Cache"):
                X_train = scipy.sparse.load_npz(train_path)
                X_test = scipy.sparse.load_npz(test_path)
                return X_train, X_test

        # 2. Generate from Scratch
        print("Generating Behavioral features from scratch...")
        with Timer("Behavioral Feature Generation"):
            # Initialize Vectorizer for Community/Behavioral data
            vectorizer = TfidfVectorizer(**Config.COMMUNITY_VECTORIZER_PARAMS)

            # Fit on Train, Transform Train & Test
            X_train = vectorizer.fit_transform(self.train_df["subreddit_text"])
            X_test = vectorizer.transform(self.test_df["subreddit_text"])

            # 3. Save to Cache
            scipy.sparse.save_npz(train_path, X_train)
            scipy.sparse.save_npz(test_path, X_test)

        return X_train, X_test

    def get_semantic_features(self, load_cached_data: bool = True):
        """
        Generates Dense Semantic Features using Sentence Transformers.
        Embeddings are standardized using StandardScaler.

        Key Configuration:
            - Model: all-MiniLM-L6-v2

        Returns:
            tuple: (X_train_semantic, X_test_semantic) as numpy arrays.
        """
        train_path = os.path.join(self.cache_dir, "X_train_semantic.npy")
        test_path = os.path.join(self.cache_dir, "X_test_semantic.npy")

        # 1. Attempt Load from Cache
        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Semantic features...")
            with Timer("Load Semantic Cache"):
                X_train = np.load(train_path)
                X_test = np.load(test_path)
                return X_train, X_test

        # 2. Generate from Scratch
        print("Generating Semantic features from scratch...")
        with Timer("Semantic Feature Generation"):
            # Load Model
            model = SentenceTransformer(Config.EMBEDDING_MODEL)

            # Encode Texts (batch_size for efficiency)
            train_texts = self.train_df["text_combined"].tolist()
            test_texts = self.test_df["text_combined"].tolist()

            X_train_raw = model.encode(
                train_texts, show_progress_bar=False, batch_size=32
            )
            X_test_raw = model.encode(
                test_texts, show_progress_bar=False, batch_size=32
            )

            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # 3. Save to Cache
            np.save(train_path, X_train)
            np.save(test_path, X_test)

        return X_train, X_test

    def get_metadata_features(self, load_cached_data: bool = True):
        """
        Generates Dense Metadata Features from allow-listed columns.
        Features are standardized using StandardScaler.

        Returns:
            tuple: (X_train_meta, X_test_meta) as numpy arrays.
        """
        train_path = os.path.join(self.cache_dir, "X_train_meta.npy")
        test_path = os.path.join(self.cache_dir, "X_test_meta.npy")

        # 1. Attempt Load from Cache
        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Metadata features...")
            with Timer("Load Metadata Cache"):
                X_train = np.load(train_path)
                X_test = np.load(test_path)
                return X_train, X_test

        # 2. Generate from Scratch
        print("Generating Metadata features from scratch...")
        with Timer("Metadata Feature Generation"):
            # Extract numerical columns (already imputed by DataFactory)
            X_train_raw = self.train_df[Config.METADATA_COLS].values
            X_test_raw = self.test_df[Config.METADATA_COLS].values

            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # 3. Save to Cache
            np.save(train_path, X_train)
            np.save(test_path, X_test)

        return X_train, X_test
