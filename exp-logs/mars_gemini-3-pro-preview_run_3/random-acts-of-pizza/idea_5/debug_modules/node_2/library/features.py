import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    WORKING_DIR,
    SEED,
    TEXT_COL,
    SUBREDDIT_COL,
    NUMERICAL_COLS,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    SVD_COMPONENTS,
    SBERT_MODEL,
)
from library.utils import set_seed, timer


class FeatureEngineer:
    def __init__(self, load_from_cache=True, debug=False):
        """
        Initializes the FeatureEngineer.

        Args:
            load_from_cache (bool): Whether to load processed features from cache if available.
            debug (bool): Whether running in debug mode (affects cache filenames).
        """
        self.load_from_cache = load_from_cache
        self.debug = debug
        self.cache_dir = os.path.join(WORKING_DIR, "features")
        os.makedirs(self.cache_dir, exist_ok=True)

        set_seed(SEED)

        # Initialize Transformers
        # 1. Lexical Transformer
        self.lexical_tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )

        # 2. Community Transformers
        # Adjust min_df for debug mode to avoid empty vocabulary on small samples
        min_df = 1 if self.debug else 2
        self.community_tfidf = TfidfVectorizer(min_df=min_df, max_features=10000)
        self.community_svd = TruncatedSVD(
            n_components=SVD_COMPONENTS, random_state=SEED
        )

        # 3. Metadata Imputer
        self.imputer = SimpleImputer(strategy="median")

        # 4. Semantic Model (Lazy loaded in transform/fit to save resources if cached)
        self.sbert_model = None

    def _get_sbert_model(self):
        if self.sbert_model is None:
            # We use CPU or GPU automatically
            device = (
                "cuda"
                if scipy.sparse.issparse(np.array([1])) is False
                and os.system("nvidia-smi") == 0
                else None
            )
            self.sbert_model = SentenceTransformer(SBERT_MODEL)
        return self.sbert_model

    def _process_subreddits(self, series):
        """Converts list of subreddits to space-separated string."""
        return series.apply(lambda x: " ".join(x) if isinstance(x, list) else "")

    def _process_text(self, series):
        """Handles NaN in text."""
        return series.fillna("").astype(str)

    def fit(self, X_train):
        """
        Fits the internal transformers on the training data.

        Args:
            X_train (pd.DataFrame): Training features.
        """
        print("Fitting feature transformers...")

        # 1. Metadata
        print("  Fitting Metadata Imputer...")
        self.imputer.fit(X_train[NUMERICAL_COLS])

        # 2. Lexical
        print("  Fitting Lexical TF-IDF...")
        text_data = self._process_text(X_train[TEXT_COL])
        self.lexical_tfidf.fit(text_data)

        # 3. Community
        print("  Fitting Community TF-IDF & SVD...")
        subreddit_text = self._process_subreddits(X_train[SUBREDDIT_COL])
        subreddit_tfidf = self.community_tfidf.fit_transform(subreddit_text)
        self.community_svd.fit(subreddit_tfidf)

        # 4. Semantic
        # SBERT is pre-trained, no fitting needed on our corpus for this approach.
        # However, we ensure the model is loaded.
        _ = self._get_sbert_model()

        print("Fitting complete.")
        return self

    def transform(self, X, split_name):
        """
        Transforms the data into three views: Lexical, Semantic, and Community.
        Handles caching based on split_name.

        Args:
            X (pd.DataFrame): Data to transform.
            split_name (str): Name of the split (e.g., 'train', 'val', 'test') for caching.

        Returns:
            dict: Dictionary containing 'lexical' (sparse), 'semantic' (dense), 'community' (dense) matrices.
        """
        prefix = "debug_" if self.debug else ""

        # Define cache paths
        path_lexical = os.path.join(
            self.cache_dir, f"{prefix}X_{split_name}_lexical.npz"
        )
        path_semantic = os.path.join(
            self.cache_dir, f"{prefix}X_{split_name}_semantic.npy"
        )
        path_community = os.path.join(
            self.cache_dir, f"{prefix}X_{split_name}_community.npy"
        )

        # Check cache
        if self.load_from_cache:
            if (
                os.path.exists(path_lexical)
                and os.path.exists(path_semantic)
                and os.path.exists(path_community)
            ):
                print(f"Loading {split_name} features from cache...")
                return {
                    "lexical": scipy.sparse.load_npz(path_lexical),
                    "semantic": np.load(path_semantic),
                    "community": np.load(path_community),
                }

        print(f"Generating features for {split_name}...")

        # --- 0. Common Metadata Features ---
        # These are concatenated to every view
        meta_features = self.imputer.transform(X[NUMERICAL_COLS])
        # Ensure float32 for efficiency
        meta_features = meta_features.astype(np.float32)

        # --- 1. Lexical View (Sparse Text + Dense Meta) ---
        with timer(f"Lexical Feature Generation ({split_name})"):
            text_data = self._process_text(X[TEXT_COL])
            lexical_tfidf = self.lexical_tfidf.transform(text_data)
            # Concatenate sparse TFIDF with dense metadata
            # We convert metadata to sparse for efficient hstack, result is sparse CSR
            meta_sparse = scipy.sparse.csr_matrix(meta_features)
            X_lexical = scipy.sparse.hstack([lexical_tfidf, meta_sparse]).tocsr()

        # --- 2. Semantic View (Dense Embeddings + Dense Meta) ---
        with timer(f"Semantic Feature Generation ({split_name})"):
            text_data_list = self._process_text(X[TEXT_COL]).tolist()
            model = self._get_sbert_model()
            # Encode returns numpy array
            embeddings = model.encode(
                text_data_list,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            X_semantic = np.hstack([embeddings, meta_features]).astype(np.float32)

        # --- 3. Community View (Dense SVD + Dense Meta) ---
        with timer(f"Community Feature Generation ({split_name})"):
            subreddit_text = self._process_subreddits(X[SUBREDDIT_COL])
            subreddit_tfidf = self.community_tfidf.transform(subreddit_text)
            subreddit_svd = self.community_svd.transform(subreddit_tfidf)
            X_community = np.hstack([subreddit_svd, meta_features]).astype(np.float32)

        # --- Save to Cache ---
        print(f"Caching features to {self.cache_dir}...")
        scipy.sparse.save_npz(path_lexical, X_lexical)
        np.save(path_semantic, X_semantic)
        np.save(path_community, X_community)

        return {"lexical": X_lexical, "semantic": X_semantic, "community": X_community}

    def fit_transform(self, X_train, split_name="train"):
        """
        Fits on X_train and transforms it.
        """
        # If cache exists and we want to load, we can skip fit technically,
        # but to be safe and consistent with the API, we check cache inside transform.
        # However, if we don't have cache, we MUST fit.

        # Check if cache exists to potentially skip fitting logic (optimization)
        prefix = "debug_" if self.debug else ""
        path_lexical = os.path.join(
            self.cache_dir, f"{prefix}X_{split_name}_lexical.npz"
        )

        if self.load_from_cache and os.path.exists(path_lexical):
            # If cache exists, we assume transformers were fitted in a previous run
            # OR we just load the data and don't care about the transformer state
            # (unless we need to transform test set later).
            # To ensure transformers are ready for Test set, we should fit.
            # But fitting is fast for TFIDF/SVD.
            self.fit(X_train)
            return self.transform(X_train, split_name)
        else:
            self.fit(X_train)
            return self.transform(X_train, split_name)
