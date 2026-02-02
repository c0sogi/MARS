import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import save_to_cache, load_from_cache, print_header, set_seed, timer


class FeaturePipeline:
    """
    FeaturePipeline generates the four specific feature sets required by the Hept-View architecture.
    It handles text concatenation, vectorization, embedding generation, and metadata preprocessing.
    It enforces strict caching to disk using .npy format to optimize runtime.
    """

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Initialize the pipeline with Union Train and Test datasets.

        Args:
            df_train: The union of training and validation data.
            df_test: The test data.
        """
        self.df_train = df_train
        self.df_test = df_test
        self.cache_dir = Config.WORKING_DIR
        set_seed(Config.SEED)

    def _get_cache_path(self, name: str, split: str) -> str:
        """Helper to generate consistent cache file paths."""
        return os.path.join(self.cache_dir, f"{name}_{split}.npy")

    def _combine_text_cols(self, df: pd.DataFrame) -> list:
        """Concatenates title and body text for processing."""
        return (
            df[Config.TEXT_COLS]
            .apply(lambda x: " ".join(x.astype(str)), axis=1)
            .tolist()
        )

    def _process_subreddits(self, df: pd.DataFrame) -> list:
        """Converts subreddit lists into space-separated strings for TF-IDF."""
        col = Config.SUBREDDIT_COL
        if col not in df.columns:
            return [""] * len(df)

        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            elif isinstance(x, np.ndarray):
                return " ".join(x.tolist())
            return str(x) if x else ""

        return df[col].apply(join_subs).tolist()

    def get_augmented_metadata(self, load_cached_data: bool = True):
        """
        Generates the Augmented Global Metadata feature set.
        Includes raw timestamps, restored RAOP history, and user stats.
        Applies Median Imputation and Standard Scaling.
        """
        print_header("Generating Augmented Metadata")
        train_path = self._get_cache_path("metadata", "train")
        test_path = self._get_cache_path("metadata", "test")

        if load_cached_data:
            X_train = load_from_cache(train_path)
            X_test = load_from_cache(test_path)
            if X_train is not None and X_test is not None:
                print("Loaded metadata from cache.")
                return X_train, X_test

        # Select allow-listed columns
        cols = Config.METADATA_ALLOW_LIST

        # Extract and coerce to numeric (handling any potential formatting issues)
        train_data = self.df_train[cols].apply(pd.to_numeric, errors="coerce")
        test_data = self.df_test[cols].apply(pd.to_numeric, errors="coerce")

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train_data)
        X_test = imputer.transform(test_data)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Cache results
        save_to_cache(X_train, train_path)
        save_to_cache(X_test, test_path)

        return X_train, X_test

    def get_granular_lexical(self, load_cached_data: bool = True):
        """
        Generates Granular Lexical Features using TF-IDF.
        Uses a custom token pattern to capture agency ('I') and currency symbols.
        """
        print_header("Generating Granular Lexical Features (TF-IDF)")
        train_path = self._get_cache_path("lexical", "train")
        test_path = self._get_cache_path("lexical", "test")

        if load_cached_data:
            X_train = load_from_cache(train_path)
            X_test = load_from_cache(test_path)
            if X_train is not None and X_test is not None:
                print("Loaded lexical features from cache.")
                return X_train, X_test

        text_train = self._combine_text_cols(self.df_train)
        text_test = self._combine_text_cols(self.df_test)

        # Initialize Vectorizer with granular config
        vectorizer = TfidfVectorizer(**Config.TFIDF_TEXT_PARAMS)

        # Fit on train, transform both
        X_train_sparse = vectorizer.fit_transform(text_train)
        X_test_sparse = vectorizer.transform(text_test)

        # Convert to dense for .npy storage (safe due to max_features constraint)
        X_train = X_train_sparse.toarray().astype(np.float32)
        X_test = X_test_sparse.toarray().astype(np.float32)

        save_to_cache(X_train, train_path)
        save_to_cache(X_test, test_path)

        return X_train, X_test

    def get_behavioral_sparse(self, load_cached_data: bool = True):
        """
        Generates Behavioral Features using Bag-of-Concepts TF-IDF on subreddit history.
        """
        print_header("Generating Behavioral Features (Community TF-IDF)")
        train_path = self._get_cache_path("behavioral", "train")
        test_path = self._get_cache_path("behavioral", "test")

        if load_cached_data:
            X_train = load_from_cache(train_path)
            X_test = load_from_cache(test_path)
            if X_train is not None and X_test is not None:
                print("Loaded behavioral features from cache.")
                return X_train, X_test

        # Process subreddit lists into strings
        subs_train = self._process_subreddits(self.df_train)
        subs_test = self._process_subreddits(self.df_test)

        vectorizer = TfidfVectorizer(**Config.TFIDF_COMMUNITY_PARAMS)

        X_train_sparse = vectorizer.fit_transform(subs_train)
        X_test_sparse = vectorizer.transform(subs_test)

        # Convert to dense
        X_train = X_train_sparse.toarray().astype(np.float32)
        X_test = X_test_sparse.toarray().astype(np.float32)

        save_to_cache(X_train, train_path)
        save_to_cache(X_test, test_path)

        return X_train, X_test

    def get_semantic_dense(self, load_cached_data: bool = True):
        """
        Generates Semantic Dense Features using sentence-transformers embeddings.
        """
        print_header("Generating Semantic Dense Features (Embeddings)")
        train_path = self._get_cache_path("semantic", "train")
        test_path = self._get_cache_path("semantic", "test")

        if load_cached_data:
            X_train = load_from_cache(train_path)
            X_test = load_from_cache(test_path)
            if X_train is not None and X_test is not None:
                print("Loaded semantic features from cache.")
                return X_train, X_test

        text_train = self._combine_text_cols(self.df_train)
        text_test = self._combine_text_cols(self.df_test)

        # Load embedding model
        model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Encode with timing
        with timer("Encoding Train"):
            X_train = model.encode(
                text_train,
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        with timer("Encoding Test"):
            X_test = model.encode(
                text_test,
                batch_size=Config.BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        save_to_cache(X_train, train_path)
        save_to_cache(X_test, test_path)

        return X_train, X_test
